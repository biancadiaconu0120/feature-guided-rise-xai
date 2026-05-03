import builtins

# =========================
# FILTER TERMINAL SPAM
# =========================
_original_print = builtins.print

def filtered_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)

    hide = [
        "Setting up PyTorch plugin",
        "upfirdn2d_plugin",
        "Evaluating images:",
        "Processing:",
    ]

    if any(text in msg for text in hide):
        return

    _original_print(*args, **kwargs)

builtins.print = filtered_print


import os
import warnings

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

import sys

# =========================
# FIX IMPORT PATH (VERY IMPORTANT FOR RISE)
# =========================
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


# =========================
# LOGGER
# =========================
class Log:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    @staticmethod
    def info(msg):
        print(f"{Log.BLUE}[INFO]{Log.RESET} {msg}")

    @staticmethod
    def success(msg):
        print(f"{Log.GREEN}[OK]{Log.RESET} {msg}")

    @staticmethod
    def warn(msg):
        print(f"{Log.YELLOW}[WARN]{Log.RESET} {msg}")

    @staticmethod
    def error(msg):
        print(f"{Log.RED}[ERROR]{Log.RESET} {msg}")

    @staticmethod
    def debug(msg):
        print(f"{Log.CYAN}[DEBUG]{Log.RESET} {msg}")

    @staticmethod
    def section(title):
        print(f"\n{Log.BOLD}==== {title} ===={Log.RESET}")


# =========================
# PATHS
# =========================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

STYLEGAN_PATH = os.path.join(PROJECT_ROOT, "stylegan2-ada-pytorch")

print("DEBUG StyleGAN path:", STYLEGAN_PATH)

sys.path.insert(0, STYLEGAN_PATH)

from utils import load_discriminator

from grad_cam import (
    baseline_gradcam,
    multilayer_gradcam,
    guided_gradcam,
    feature_fusion_gradcam,
    combined_supermap,
    _img_tensor_to_display_np,
    save_overlay_and_raw
)

try:
    import rise
    from rise import generate_rise

    Log.success(f"Using RISE file: {rise.__file__}")
    has_rise = True
except Exception as e:
    Log.error(f"Could not import rise.py: {repr(e)}")
    has_rise = False


# =========================
# CONFIG
# =========================
IMG_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Log.info(f"Device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class DiscriminatorTwoClassWrapper(torch.nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, x):
        logit = self.discriminator(x, None)
        out = torch.cat([-logit, logit], dim=1)
        return torch.nn.functional.softmax(out, dim=1)


correct = 0
total = 0
all_labels = []
all_scores = []


def classify_and_explain(image_path, D_raw, D_softmax, target_layer, multilayer_list, result_root, show_images=False):
    global correct, total, all_labels, all_scores

    base = os.path.splitext(os.path.basename(image_path))[0]
    Log.section(f"Processing {base}")

    gradcam_root = os.path.join(result_root, "gradcam")
    folders = {
        "baseline": os.path.join(gradcam_root, "baseline"),
        "multilayer": os.path.join(gradcam_root, "multilayer"),
        "guided": os.path.join(gradcam_root, "guided"),
        "feature_fusion": os.path.join(gradcam_root, "feature_fusion"),
        "combined": os.path.join(gradcam_root, "combined")
    }

    for p in folders.values():
        os.makedirs(p, exist_ok=True)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        Log.error(f"Could not load image {image_path}: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True)

    probs = D_softmax(img_tensor)[0].detach().cpu().numpy()
    label_idx = int(probs.argmax())
    label = "Real" if label_idx == 1 else "Fake"
    score = float(probs[label_idx])

    Log.info(f"Prediction: {label} | Score: {score:.4f}")

    filename_lower = os.path.basename(image_path).lower()
    true_label = 1 if "real" in filename_lower else 0

    all_labels.append(true_label)
    all_scores.append(score)
    total += 1

    if label_idx == true_label:
        correct += 1

    img_np = _img_tensor_to_display_np(img_tensor)

    # =========================
    # GRAD-CAM
    # =========================
    Log.section("Grad-CAM")

    try:
        cam_baseline = baseline_gradcam(
            D_softmax, img_tensor, target_layer,
            class_idx=label_idx,
            upsample_size=(IMG_SIZE, IMG_SIZE),
            device=DEVICE
        )
        save_overlay_and_raw(folders["baseline"], base, img_np, cam_baseline)
        Log.success("Saved baseline Grad-CAM")
    except Exception as e:
        Log.error(f"Baseline Grad-CAM failed: {e}")

    # =========================
    # RISE + SIFT
    # =========================
    if has_rise:
        Log.section("RISE")

        try:
            img_np_raw = np.array(img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))

            Log.debug(f"Original resized image shape: {img_np_raw.shape}")
            Log.debug(f"Input tensor shape: {tuple(img_tensor.shape)}")

            rise_maps = generate_rise(
                model=D_softmax,
                input_tensor=img_tensor,
                class_idx=label_idx,
                result_root=result_root,
                base_name=base,
                orig_img=img_np_raw,
                variants=["baseline", "sift_only"],  # ONLY 2 METHODS
                n_masks=500,
                batch_size=4,
                mask_size=16,
                device=DEVICE
            )

            Log.success(f"Generated RISE methods: {list(rise_maps.keys())}")

        except Exception as e:
            Log.error(f"RISE failed: {repr(e)}")

    Log.success(f"Finished image: {base}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="stylegan2-ada-pytorch/FFHQ.pkl")
    parser.add_argument("--folder", default="images/test")
    parser.add_argument("--results", default="results")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    Log.section("Loading model")

    D_raw = load_discriminator(args.network)
    D_softmax = DiscriminatorTwoClassWrapper(D_raw).to(DEVICE).eval()

    Log.success(f"Discriminator loaded from: {args.network}")

    image_files = [
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    Log.info(f"Found {len(image_files)} images")

    multilayer_list = ["b64.conv1", "b32.conv1", "b16.conv1"]
    baseline_layer = "b64.conv1"

    for img_path in image_files:  # REMOVED tqdm
        classify_and_explain(
            img_path,
            D_raw,
            D_softmax,
            target_layer=baseline_layer,
            multilayer_list=multilayer_list,
            result_root=args.results,
            show_images=not args.no_show
        )