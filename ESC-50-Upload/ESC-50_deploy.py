import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import librosa
import numpy as np
import pandas as pd
import tempfile

# Common Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load ESC-50 Metadata
@st.cache_data
def load_class_names():
    df = pd.read_csv("ESC-50-master/meta/esc50.csv")
    return df.drop_duplicates("target").sort_values("target")["category"].tolist()


class_names = load_class_names()


# Model 1: EPASS
class ResNet1DAudioEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.projection(x)
        return x


class EPASSProjectors(nn.Module):
    def __init__(self, embed_dim=256, proj_dim=128, num_proj=3):
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, proj_dim),
                    nn.ReLU(),
                    nn.Linear(proj_dim, proj_dim),
                )
                for _ in range(num_proj)
            ]
        )

    def forward(self, x):
        projections = [proj(x) for proj in self.projectors]
        ensemble = torch.stack(projections).mean(dim=0)
        return ensemble, projections


@st.cache_resource
def load_epass():
    encoder = ResNet1DAudioEncoder().to(device)
    projectors = EPASSProjectors().to(device)
    classifier = nn.Linear(256, 50).to(device)
    checkpoint = torch.load(
        "models/ESC-50/EPASS/best_model_fold1.pt",
        map_location=device,
        weights_only=False,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    projectors.load_state_dict(checkpoint["projectors_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    encoder.eval()
    classifier.eval()
    return encoder, classifier


def predict_epass(audio_path):
    encoder, classifier = load_epass()
    y, sr = librosa.load(audio_path, sr=44100)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = (
        torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        features = encoder(mel_tensor)
        output = classifier(features)
        pred = torch.argmax(output, dim=1).item()
    return pred, output.softmax(dim=1).squeeze().cpu().numpy()


# Model 2: FlatMatch
class ESC50CNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)


@st.cache_resource
def load_flatmatch():
    model = ESC50CNN().to(device)
    checkpoint = torch.load(
        "models/ESC-50/FlatMatchnew/best_model_fold3.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_flatmatch(audio_path):
    model = load_flatmatch()
    y, sr = librosa.load(audio_path, sr=44100)
    y = np.pad(y, (0, max(0, sr * 5 - len(y))))[: sr * 5]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mel_tensor = F.interpolate(mel_tensor, size=(224, 224), mode="bilinear").to(device)
    with torch.no_grad():
        output = model(mel_tensor)
        probs = output.softmax(dim=1).squeeze().cpu().numpy()
        pred = np.argmax(probs)
    return pred, probs


# Model 3: OSP
class OSPModel(nn.Module):
    def __init__(self, num_classes=40, backbone="resnet18"):
        super(OSPModel, self).__init__()
        self.encoder = self._build_encoder(backbone)

        with torch.no_grad():
            dummy = torch.randn(1, 1, 128, 431)
            feature_size = self.encoder(dummy).shape[1]

        self.classifier_head = nn.Linear(feature_size, num_classes)
        self.ssp_head = nn.Linear(feature_size, 4)  # 4 audio transforms

    def _build_encoder(self, backbone):
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity()
        return model

    def forward(self, x, task="class"):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)

        if task == "class":
            return self.classifier_head(features)
        elif task == "ssp":
            return self.ssp_head(features)
        elif task == "feature":
            return features
        else:
            raise ValueError(f"Unknown task '{task}'")


osp_label_map = {
    v: k
    for k, v in {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        10: 8,
        12: 9,
        13: 10,
        14: 11,
        15: 12,
        17: 13,
        18: 14,
        20: 15,
        21: 16,
        22: 17,
        24: 18,
        27: 19,
        28: 20,
        30: 21,
        31: 22,
        32: 23,
        33: 24,
        34: 25,
        35: 26,
        36: 27,
        37: 28,
        38: 29,
        39: 30,
        40: 31,
        41: 32,
        42: 33,
        43: 34,
        44: 35,
        46: 36,
        47: 37,
        48: 38,
        49: 39,
    }.items()
}


@st.cache_resource
def load_osp():
    model = OSPModel(num_classes=len(osp_label_map))
    checkpoint = torch.load("models/ESC-50/OSP/pretrained_sor.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_osp(audio_path):
    model = load_osp()
    y, _ = librosa.load(audio_path, sr=44100)
    y = np.pad(y, (0, max(0, 44100 * 5 - len(y))))[: 44100 * 5]
    mel = librosa.feature.melspectrogram(y=y, sr=44100, n_mels=128)
    mel_db = librosa.power_to_db(mel + 1e-6, ref=np.max)
    mel_db = np.clip(mel_db, -80, 0)
    mel_tensor = (
        torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        logits = model(mel_tensor)
        pred = logits.argmax(dim=1).item()
    return osp_label_map[pred]


# Streamlit UI
st.title("ESC-50 Audio Classification (Compare Models)")

uploaded_file = st.file_uploader("Drop a .wav/.mp3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.subheader("Predictions")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### EPASS")
        pred_epass, probs_epass = predict_epass(tmp_path)
        st.write(f"Class ID: {pred_epass}")
        st.write(f"Label: {class_names[pred_epass]}")

    with col2:
        st.markdown("### FlatMatch")
        pred_flat, probs_flat = predict_flatmatch(tmp_path)
        st.write(f"Class ID: {pred_flat}")
        st.write(f"Label: {class_names[pred_flat]}")
        st.bar_chart(probs_flat)

    with col3:
        st.markdown("### OSP")
        pred_osp = predict_osp(tmp_path)
        st.write(f"Mapped Class ID: {pred_osp}")
        st.write(f"Label: {class_names[pred_osp]}")
