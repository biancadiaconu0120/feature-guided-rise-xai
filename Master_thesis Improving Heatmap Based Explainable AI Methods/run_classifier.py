import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stylegan2-ada-pytorch')))
sys.path.append("../stylegan2-ada-pytorch")

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
    from rise import generate_rise

    has_rise = True
except Exception as e:
    print("Warning: could not import rise.py:", repr(e))
    has_rise = False

IMG_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Fehler beim Laden von {image_path}: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True)

    probs = D_softmax(img_tensor)[0].detach().cpu().numpy()
    label_idx = int(probs.argmax())
    label = "Real" if label_idx == 1 else "Fake"
    score = float(probs[label_idx])

    print(f"{os.path.basename(image_path)} --> Score: {score:.4f} ({label})")

    filename_lower = os.path.basename(image_path).lower()
    true_label = 1 if "real" in filename_lower else 0

    all_labels.append(true_label)
    all_scores.append(score)
    total += 1
    if label_idx == true_label:
        correct += 1

    img_np = _img_tensor_to_display_np(img_tensor)

    try:
        cam_baseline = baseline_gradcam(
            D_softmax,
            img_tensor,
            target_layer,
            class_idx=label_idx,
            upsample_size=(IMG_SIZE, IMG_SIZE),
            device=DEVICE
        )
        save_overlay_and_raw(folders['baseline'], base, img_np, cam_baseline)
    except Exception as e:
        print("Fehler Baseline Grad-CAM:", e)

    try:
        cam_multilayer = multilayer_gradcam(
            D_softmax,
            img_tensor,
            multilayer_list,
            class_idx=label_idx,
            upsample_size=(IMG_SIZE, IMG_SIZE),
            layer_weights=None,
            device=DEVICE
        )
        save_overlay_and_raw(folders['multilayer'], base, img_np, cam_multilayer)
    except Exception as e:
        print("Fehler Multi-layer Grad-CAM:", e)

    try:
        guided_rgb = guided_gradcam(
            D_softmax,
            img_tensor,
            target_layer,
            class_idx=label_idx,
            upsample_size=(IMG_SIZE, IMG_SIZE),
            device=DEVICE
        )
        save_overlay_and_raw(folders['guided'], base, img_np, guided_rgb)
    except Exception as e:
        print("Fehler Guided Grad-CAM:", e)

    try:
        cam_feat = feature_fusion_gradcam(
            D_softmax,
            img_tensor,
            multilayer_list,
            class_idx=label_idx,
            upsample_size=(IMG_SIZE, IMG_SIZE),
            layer_weights=None,
            kaze_params={'n_keypoints': 300, 'blob_radius': 12},
            device=DEVICE
        )
        save_overlay_and_raw(folders['feature_fusion'], base, img_np, cam_feat)
    except Exception as e:
        print("Fehler Feature-Fusion Grad-CAM:", e)

    try:
        combined = combined_supermap(
            D_softmax,
            img_tensor,
            baseline_layer=target_layer,
            multilayer_names=multilayer_list,
            class_idx=label_idx,
            upsample_size=(IMG_SIZE, IMG_SIZE),
            layer_weights=None,
            kaze_params={'n_keypoints': 300, 'blob_radius': 12},
            device=DEVICE
        )
        save_overlay_and_raw(folders['combined'], base, img_np, combined)
    except Exception as e:
        print("Fehler Combined Supermap:", e)

    if has_rise:
        result_root_rise = result_root
        os.makedirs(result_root_rise, exist_ok=True)

        try:
            # IMPORTANT: keep original image aligned with network input size
            img_np_raw = np.array(img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))

            print("DEBUG: run_classifier orig resized shape:", img_np_raw.shape)
            print("DEBUG: run_classifier input_tensor shape:", tuple(img_tensor.shape))

            generate_rise(
                model=D_softmax,
                input_tensor=img_tensor,
                class_idx=label_idx,
                result_root=result_root_rise,
                base_name=base,
                orig_img=img_np_raw,
                variants=['baseline', 'sift'],
                n_masks=600,
                batch_size=4,
                mask_size=16,
                device=DEVICE
            )

        except Exception as e:
            print("RISE konnte nicht erzeugt werden:", repr(e))

    print(f"✓ Ergebnisse gespeichert (Grad-CAM + RISE Varianten) für {base}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="stylegan2-ada-pytorch/FFHQ.pkl", help="Pfad zum Discriminator .pkl")
    parser.add_argument("--folder", default="images/test", help="Ordner mit Testbildern")
    parser.add_argument("--results", default="results", help="Ordner zum Speichern der Ergebnisse")
    parser.add_argument("--no_show", action="store_true", help="Visualisierungen nicht anzeigen")
    args = parser.parse_args()

    D_raw = load_discriminator(args.network)
    D_softmax = DiscriminatorTwoClassWrapper(D_raw).to(DEVICE).eval()

    print(f"\nDiscriminator geladen aus {args.network}\n")

    test_folder = args.folder
    image_files = [
        os.path.join(test_folder, f)
        for f in os.listdir(test_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"{len(image_files)} Bilder gefunden. Starte Klassifizierung...\n")

    multilayer_list = ['b64.conv1', 'b32.conv1', 'b16.conv1']
    baseline_layer = 'b64.conv1'

    for img_path in tqdm(image_files, desc="Verarbeitung", unit="Bild"):
        classify_and_explain(
            img_path,
            D_raw,
            D_softmax,
            target_layer=baseline_layer,
            multilayer_list=multilayer_list,
            result_root=args.results,
            show_images=not args.no_show
        )

    if total > 0:
        acc = (correct / total) * 100
        print(f"\n===> Accuracy: {correct}/{total} = {acc:.2f}%")

        os.makedirs(args.results, exist_ok=True)
        with open(os.path.join(args.results, "accuracy.txt"), "w") as f:
            f.write(f"Accuracy: {correct}/{total} = {acc:.2f}%\n")

        all_labels_np = np.array(all_labels)
        all_scores_np = np.array(all_scores)

        try:
            fpr, tpr, thresholds_roc = roc_curve(all_labels_np, all_scores_np)
            roc_auc = auc(fpr, tpr)
            optimal_idx_roc = np.argmax(tpr - fpr)
            optimal_thresh_roc = thresholds_roc[optimal_idx_roc]

            plt.figure(figsize=(3.5, 2.6))
            plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.scatter(fpr[optimal_idx_roc], tpr[optimal_idx_roc], color='red', s=15)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC-Kurve')
            plt.tight_layout()
            plt.savefig(os.path.join(args.results, "roc_curve_plot.pdf"), dpi=300, bbox_inches="tight")
            print(f"✓ ROC-Kurve gespeichert: {os.path.join(args.results, 'roc_curve_plot.pdf')}")
        except Exception as e:
            print("ROC konnte nicht berechnet werden:", e)
    else:
        print("Keine gültigen Labels zur Accuracy- und ROC-Berechnung gefunden.")

    from evaluation_rise import evaluate_all as evaluate_rise
    from evaluation import evaluate_all as evaluate_gradcam

    evaluate_gradcam(
        results_root=args.results,
        image_folder=args.folder,
        network=args.network,
        gt_path=None
    )

    evaluate_rise(
        results_root=args.results,
        image_folder=args.folder,
        network=args.network,
        gt_path=None
    )
