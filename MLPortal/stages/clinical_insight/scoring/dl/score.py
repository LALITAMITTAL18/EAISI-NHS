"""Azure ML scoring script for the NHS Knee EfficientNet-B4 DL model.

The model is a PyTorch checkpoint saved as:
  knee_kl_efficientnet_b4_v3_final.pth

Input JSON schema:
  { "image_base64": "<base64-encoded JPEG or PNG>" }

Output JSON schema:
  { "kl_grade": 2,
    "confidence": 0.84,
    "class_probabilities": [0.02, 0.05, 0.84, 0.07, 0.02],
    "class_names": ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"] }
"""

import base64
import io
import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)
model = None
class_names = None
img_size = 224
device = None


def init():
    global model, class_names, img_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    checkpoint_path = None
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pth") or f.endswith(".pt"):
                checkpoint_path = os.path.join(root, f)
                break
        if checkpoint_path:
            break

    if checkpoint_path is None:
        raise FileNotFoundError(f"No .pth checkpoint found under {model_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", 5)
    img_size = checkpoint.get("img_size", 224)
    class_names = checkpoint.get(
        "class_names", [f"Grade {i}" for i in range(num_classes)]
    )

    from torchvision import models as tv_models

    backbone_name = checkpoint.get("architecture", "efficientnet_b4")
    backbone_fn = getattr(tv_models, backbone_name, tv_models.efficientnet_b4)
    net = backbone_fn(weights=None)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device).eval()

    model = net
    logger.info("DL model loaded from %s (device=%s)", checkpoint_path, device)


def _apply_clahe(img):
    """CLAHE preprocessing matching the training pipeline."""
    import cv2
    from PIL import Image

    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    rgb = np.stack([eq, eq, eq], axis=-1)
    return Image.fromarray(rgb)


def run(raw_data: str) -> str:
    import torchvision.transforms as T
    from PIL import Image

    data = json.loads(raw_data)
    img_bytes = base64.b64decode(data["image_base64"])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    transform = T.Compose(
        [
            T.Lambda(_apply_clahe),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted = int(probs.argmax())
    confidence = float(probs.max())

    return json.dumps(
        {
            "kl_grade": predicted,
            "confidence": confidence,
            "class_probabilities": probs.tolist(),
            "class_names": class_names,
        }
    )
