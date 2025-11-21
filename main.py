import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
import re

MODEL_PATH = "model.pth"
IMAGE_SIZE = 384
CLASS_NAMES = [
    "CrushedStandingRoomOnly",
    "Empty",
    "FewSeatsAvailable",
    "Full",
    "ManySeatsAvailable",
    "StandingRoomOnly"
]
DEVICE = torch.device("cpu")

st.set_page_config(page_title="Bus Occupancy Classifier", layout="wide")
st.title("Bus Occupancy Classifier")

@st.cache_resource
def load_model():
    model = timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )

    in_features = model.get_classifier().in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(CLASS_NAMES))
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model

def humanize(label: str) -> str:
    return " ".join(w.capitalize() for w in re.sub(r'(?<!^)(?=[A-Z])', ' ', label).split())


model = load_model()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE + 16),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Added webp support
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col_img, _, col_pred = st.columns([10, 1, 10])

    with col_img:
        st.header("Image")
        st.image(img, width="stretch")

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    sorted_results = sorted(
        zip(CLASS_NAMES, probs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    top_class, top_prob = sorted_results[0]

    with col_pred:
        st.header("Prediction")

        for idx, (cls, p) in enumerate(sorted_results, start=1):
            label = f"{idx}. **{humanize(cls)}**: {p:.2%}"

            if cls == top_class:
                st.info(label)
            else:
                st.write(label)
