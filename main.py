import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform
from dataclasses import dataclass
from typing import List

CLASS_NAMES = [
    "empty",
    "many_seats_available",
    "few_seats_available",
    "standing_room_only",
    "crushed_standing_room_only",
    "full",
]
NUM_CLASSES = len(CLASS_NAMES)

@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    model_name: str
    task_type: str
    weights_path: str

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        "MobileNetV3 Small – multiclass",
        "tf_mobilenetv3_small_minimal_100.in1k",
        "multiclass",
        "weights/multiclass_mobilenetv3_small_minimal.pth",
    ),
    ModelSpec(
        "MobileNetV3 Small – ordinal classification",
        "tf_mobilenetv3_small_minimal_100.in1k",
        "ordinal_coral",
        "weights/ordinal_coral_mobilenetv3_small_minimal.pth",
    ),
    ModelSpec(
        "MobileNetV3 Small – ordinal regression",
        "tf_mobilenetv3_small_minimal_100.in1k",
        "ordinal_regression",
        "weights/ordinal_regression_mobilenetv3_small_minimal.pth",
    ),
    ModelSpec(
        "EfficientNet-B0 – multiclass",
        "efficientnet_b0.ra_in1k",
        "multiclass",
        "weights/multiclass_efficientnet_b0.pth",
    ),
    ModelSpec(
        "EfficientNet-B0 – ordinal classification",
        "efficientnet_b0.ra_in1k",
        "ordinal_coral",
        "weights/ordinal_coral_efficientnet_b0.pth",
    ),
    ModelSpec(
        "EfficientNet-B0 – ordinal regression",
        "efficientnet_b0.ra_in1k",
        "ordinal_regression",
        "weights/ordinal_regression_efficientnet_b0.pth",
    ),
    ModelSpec(
        "EfficientFormer-L1 – multiclass",
        "efficientformer_l1.snap_dist_in1k",
        "multiclass",
        "weights/multiclass_efficientformer_l1.pth",
    ),
    ModelSpec(
        "EfficientFormer-L1 – ordinal classification",
        "efficientformer_l1.snap_dist_in1k",
        "ordinal_coral",
        "weights/ordinal_coral_efficientformer_l1.pth",
    ),
    ModelSpec(
        "EfficientFormer-L1 – ordinal regression",
        "efficientformer_l1.snap_dist_in1k",
        "ordinal_regression",
        "weights/ordinal_regression_efficientformer_l1.pth",
    ),
]

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def coral_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(logits)
    b = s.size(0)
    k = NUM_CLASSES

    probs = torch.zeros((b, k), device=logits.device)
    probs[:, 0] = 1.0 - s[:, 0]

    for i in range(1, k - 1):
        probs[:, i] = s[:, i - 1] - s[:, i]

    probs[:, k - 1] = s[:, k - 2]
    probs = torch.clamp(probs, min=0.0)

    return probs / (probs.sum(dim=1, keepdim=True) + 1e-12)


def ordinal_regression_to_probs(logits: torch.Tensor, sigma: float = 0.75) -> torch.Tensor:
    v = logits.view(-1)
    idx = torch.arange(NUM_CLASSES, device=logits.device).float()
    dist2 = (v.unsqueeze(1) - idx.unsqueeze(0)) ** 2
    probs = torch.exp(-dist2 / (2 * sigma * sigma))
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)

    return probs


def build_model(spec: ModelSpec) -> torch.nn.Module:
    if spec.task_type == "multiclass":
        outputs = NUM_CLASSES
    elif spec.task_type == "ordinal_coral":
        outputs = NUM_CLASSES - 1
    elif spec.task_type == "ordinal_regression":
        outputs = 1
    else:
        raise ValueError(spec.task_type)

    model = timm.create_model(
        spec.model_name,
        pretrained=False,
        num_classes=outputs,
    )

    state_dict = torch.load(spec.weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def build_transform(model):
    cfg = resolve_data_config({}, model=model)

    return create_transform(**cfg, is_training=False)

@st.cache_resource
def load_model(spec: ModelSpec):
    model = build_model(spec)
    transform = build_transform(model)

    return model, transform

def predict(spec, model, transform, img):
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    if spec.task_type == "multiclass":
        probs = F.softmax(logits, dim=1)
    elif spec.task_type == "ordinal_coral":
        probs = coral_logits_to_class_probs(logits)
    elif spec.task_type == "ordinal_regression":
        probs = ordinal_regression_to_probs(logits)

    pred_idx = int(torch.argmax(probs, dim=1))

    return {
        "pred_idx": pred_idx,
        "pred_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[0, pred_idx]),
        "probs": probs[0].cpu().tolist(),
        "logits": logits[0].cpu().tolist(),
    }


st.set_page_config(page_title="Bus Occupancy Classifier")
st.title("Bus Occupancy Classifier")
st.subheader("Multi-Model and Training Strategy Comparison")

selected_models = st.multiselect(
    "Models",
    options=[m.display_name for m in MODEL_SPECS],
    default=[m.display_name for m in MODEL_SPECS],
)

st.divider()

uploaded = st.file_uploader(
    "Upload bus image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()

    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Image", width="stretch")

    st.divider()

    with st.spinner("Loading models..."):
        models = {}

        for spec in MODEL_SPECS:
            if spec.display_name in selected_models:
                model, transform = load_model(spec)
                models[spec.display_name] = (spec, model, transform)

    results = []

    with st.spinner("Running inference..."):
        for name in selected_models:
            spec, model, transform = models[name]
            out = predict(spec, model, transform, img)
            results.append((name, spec, out))

    st.subheader("Predictions")

    votes = [out["pred_idx"] for _, _, out in results]
    counts = {i: votes.count(i) for i in range(NUM_CLASSES)}
    best = max(counts.items(), key=lambda x: x[1])

    st.write(
        f"Majority prediction: **{CLASS_NAMES[best[0]]}** "
        f"({best[1]}/{len(votes)} models)"
    )

    for name, spec, out in results:
        with st.expander(
            f"{name} → {out['pred_class']} ({out['confidence']:.2%})",
            expanded=False,
        ):

            for cls, p in zip(CLASS_NAMES, out["probs"]):
                st.write(f"{cls}: **{p:.2%}**")
