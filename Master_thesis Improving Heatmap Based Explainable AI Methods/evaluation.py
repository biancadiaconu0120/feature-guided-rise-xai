import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from skimage.measure import label, regionprops
import cv2
import torch
from torchvision import transforms
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stylegan2-ada-pytorch')))
sys.path.append("../stylegan2-ada-pytorch")
try:
    from utils import load_discriminator
    has_utils = True
except Exception:
    has_utils = False


transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def load_heatmap(results_root, method, base):
    p = os.path.join(results_root, "gradcam", method, f"{base}_raw.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def list_basenames_from_folder(image_folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(exts)]
    bases = [os.path.splitext(f)[0] for f in files]
    return sorted(bases)


def read_gt_bboxes(gt_path):
    if gt_path is None:
        return {}
    if not os.path.exists(gt_path):
        print("GT path nicht gefunden:", gt_path)
        return {}

    if gt_path.lower().endswith(".json"):
        with open(gt_path, "r") as f:
            j = json.load(f)
        for k, v in j.items():
            j[k] = [[int(round(coord)) for coord in bb] for bb in v]
        return j

    df = pd.read_csv(gt_path, header=None)
    gt = {}
    for _, row in df.iterrows():
        base = str(row[0])
        bb = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
        gt.setdefault(base, []).append(bb)
    return gt


def load_image_tensor(image_folder, base, device):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    for e in exts:
        p = os.path.join(image_folder, base + e)
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            t = transform(img).unsqueeze(0).to(device)
            return t
    raise FileNotFoundError(f"Image {base} not found in {image_folder}")


def compose_image_with_mask(orig_tensor, mask, mode='blur'):
    device = orig_tensor.device
    inp = orig_tensor.detach().cpu().numpy()[0]

    if inp.min() < 0 or inp.max() > 1:
        vis = (inp * 0.5) + 0.5
    else:
        vis = inp.copy()

    vis = np.transpose(vis, (1, 2, 0))
    h, w, _ = vis.shape
    mask3 = np.stack([mask, mask, mask], axis=2)

    if mode == 'zero':
        composed = vis * mask3
    elif mode == 'blur':
        blurred = cv2.GaussianBlur((vis * 255).astype(np.uint8), (31, 31), 0).astype(np.float32) / 255.0
        composed = vis * mask3 + blurred * (1 - mask3)
    elif mode == 'mean':
        mean_color = vis.mean(axis=(0, 1), keepdims=True)
        composed = vis * mask3 + mean_color * (1 - mask3)
    else:
        raise ValueError("mode must be 'zero'|'blur'|'mean'")

    t = torch.from_numpy(np.transpose(composed, (2, 0, 1))).unsqueeze(0).to(device).float()
    if inp.min() < 0 or inp.max() > 1:
        t = (t - 0.5) / 0.5
    return t


def compute_order_from_map(sal_map):
    flat = sal_map.flatten()
    order = np.argsort(-flat)
    return order


def batched_forward(model, batch_t, device):
    with torch.no_grad():
        out = model(batch_t.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()


def deletion_insertion_auc(
    model,
    orig_tensor,
    sal_map,
    class_idx,
    steps=50,
    mode='deletion',
    baseline_mode='blur',
    device='cuda',
    batch_size=16
):
    device = torch.device(device)
    h, w = sal_map.shape
    order = compute_order_from_map(sal_map)
    total_pix = h * w
    step = int(np.ceil(total_pix / steps))
    fractions = []
    scores = []

    indices_list = [min(s * step, total_pix) for s in range(0, steps + 1)]
    for k in indices_list:
        if mode == 'deletion':
            removed_inds = order[:k]
            mask_flat = np.ones(total_pix, dtype=np.float32)
            mask_flat[removed_inds] = 0.0
            mask = mask_flat.reshape(h, w)
            inp = compose_image_with_mask(orig_tensor, mask, mode=baseline_mode)
        else:
            insert_inds = order[:k]
            mask_flat = np.zeros(total_pix, dtype=np.float32)
            mask_flat[insert_inds] = 1.0
            mask = mask_flat.reshape(h, w)
            inp = compose_image_with_mask(orig_tensor, mask, mode='mean')

        probs = batched_forward(model, inp, device)
        p = probs[0, class_idx]
        scores.append(p)
        fractions.append(k / float(total_pix))

    auc = float(np.trapz(scores, fractions))
    return auc, np.array(scores), np.array(fractions)


def sparsity_metrics(sal_map, threshold=0.5):
    h, w = sal_map.shape
    mask_below = (sal_map < threshold).astype(np.uint8)
    fraction_below = mask_below.sum() / float(h * w)
    labeled = label(1 - mask_below)
    n_comp = labeled.max()
    areas = [r.area for r in regionprops(labeled)] if n_comp > 0 else []
    avg_area = (np.mean(areas) / (h * w)) if areas else 0.0
    return fraction_below, int(n_comp), float(avg_area)


def iou_with_bbox(sal_map, bbox, threshold=0.5):
    h, w = sal_map.shape
    mask = (sal_map > threshold).astype(np.uint8)

    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, int(round(xmin)))
    ymin = max(0, int(round(ymin)))
    xmax = min(w - 1, int(round(xmax)))
    ymax = min(h - 1, int(round(ymax)))

    bbox_mask = np.zeros_like(mask)
    bbox_mask[ymin:ymax + 1, xmin:xmax + 1] = 1

    inter = (mask & bbox_mask).sum()
    union = (mask | bbox_mask).sum()
    return 0.0 if union == 0 else (inter / union)


def pointing_game(sal_map, bbox):
    h, w = sal_map.shape
    idx = int(np.argmax(sal_map))
    y, x = divmod(idx, w)
    xmin, ymin, xmax, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax


def aggregate_and_test(df_metrics, metric_col='deletion_auc', group_col='method', baseline='baseline'):
    methods = df_metrics[group_col].unique().tolist()
    summary = []
    pvals = {}

    for m in methods:
        arr = df_metrics[df_metrics[group_col] == m][metric_col].dropna().values
        summary.append({
            'method': m,
            'mean': float(np.mean(arr)) if arr.size > 0 else np.nan,
            'std': float(np.std(arr)) if arr.size > 0 else np.nan,
            'n': int(arr.size)
        })

        if m != baseline:
            merged = pd.merge(
                df_metrics[df_metrics[group_col] == baseline][['image', metric_col]],
                df_metrics[df_metrics[group_col] == m][['image', metric_col]],
                on='image',
                suffixes=('_base', '_m')
            )
            a = merged[metric_col + '_base'].values
            b = merged[metric_col + '_m'].values

            if a.size == 0:
                p = 1.0
            else:
                try:
                    _, p = stats.wilcoxon(a, b)
                except Exception:
                    _, p = stats.ttest_rel(a, b)

            pvals[m] = float(p)

    return pd.DataFrame(summary), pvals


def plot_boxplots(df_metrics, metric_col, out_path, title=None):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='method', y=metric_col, data=df_metrics, order=sorted(df_metrics['method'].unique()))
    plt.xticks(rotation=30)
    plt.title(title or metric_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_curves_example(curves_dict, out_path, title="Deletion/Insertion example"):
    plt.figure(figsize=(6, 4))
    for m, (scores, fracs) in curves_dict.items():
        plt.plot(fracs, scores, label=m)
    plt.xlabel('Fraction')
    plt.ylabel('Score (class prob)')
    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate_all(
    results_root,
    image_folder,
    network=None,
    gt_path=None,
    methods=None,
    steps=50,
    batch=16,
    n_examples=6,
    baseline_method='baseline',
    device=None,
    out_root=None
):
    if methods is None:
        methods = ['baseline', 'multilayer', 'guided', 'feature_fusion', 'combined']

    if out_root is None:
        out_root = os.path.join(results_root, "evaluation")

    os.makedirs(out_root, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    basenames = list_basenames_from_folder(image_folder)
    gt = read_gt_bboxes(gt_path) if gt_path else {}

    model = None
    if network:
        if not has_utils:
            print("utils.load_discriminator nicht verfuegbar. Network kann nicht geladen werden.")
        else:
            D_raw = load_discriminator(network)

            class DiscriminatorTwoClassWrapper(torch.nn.Module):
                def __init__(self, discriminator):
                    super().__init__()
                    self.discriminator = discriminator

                def forward(self, x):
                    logit = self.discriminator(x, None)
                    out = torch.cat([-logit, logit], dim=1)
                    return torch.nn.functional.softmax(out, dim=1)

            model = DiscriminatorTwoClassWrapper(D_raw).to(device).eval()
            print("Modell geladen fuer Deletion/Insertion:", network)

    rows = []
    forced_examples = {"seed0151", "seed0152"}
    example_curves = {}

    for base in tqdm(basenames, desc="Evaluating images"):
        img_tensor = None
        if model is not None:
            try:
                img_tensor = load_image_tensor(image_folder, base, device)
            except Exception as e:
                print("Could not load image tensor for", base, e)

        curves_for_example = {}

        for m in methods:
            heat = load_heatmap(results_root, m, base)
            if heat is None:
                continue

            if heat.ndim == 3:
                heat2 = heat.mean(axis=2)
            else:
                heat2 = heat

            heat2 = heat2 - heat2.min()
            if heat2.max() > 0:
                heat2 = heat2 / (heat2.max() + 1e-8)

            frac_above, n_comp, avg_area = sparsity_metrics(heat2)
            del_auc = np.nan
            ins_auc = np.nan

            if model is not None and img_tensor is not None:
                try:
                    with torch.no_grad():
                        probs = model(img_tensor)
                        class_idx = int(probs.argmax(dim=1)[0].item())

                    del_auc, del_scores, del_fracs = deletion_insertion_auc(
                        model,
                        img_tensor,
                        heat2,
                        class_idx,
                        steps=steps,
                        mode='deletion',
                        baseline_mode='blur',
                        device=device,
                        batch_size=batch
                    )
                    ins_auc, ins_scores, ins_fracs = deletion_insertion_auc(
                        model,
                        img_tensor,
                        heat2,
                        class_idx,
                        steps=steps,
                        mode='insertion',
                        baseline_mode='blur',
                        device=device,
                        batch_size=batch
                    )
                    curves_for_example[m] = (del_scores, del_fracs, ins_scores, ins_fracs)
                except Exception as e:
                    print("Error computing del/ins for", base, m, e)

            iou_mean = np.nan
            pointing_acc = np.nan
            if base in gt and len(gt[base]) > 0:
                ious = []
                points = []
                for bb in gt[base]:
                    try:
                        iou = iou_with_bbox(heat2, bb)
                        pg = pointing_game(heat2, bb)
                    except Exception:
                        iou = np.nan
                        pg = False
                    ious.append(iou)
                    points.append(1.0 if pg else 0.0)
                iou_mean = np.nanmean(ious)
                pointing_acc = np.nanmean(points)

            rows.append({
                'image': base,
                'method': m,
                'deletion_auc': del_auc,
                'insertion_auc': ins_auc,
                'fraction_below_0.5': frac_above,
                'n_components': n_comp,
                'avg_component_area': avg_area,
                'iou_mean': iou_mean,
                'pointing_acc': pointing_acc
            })

        if curves_for_example and (len(example_curves) < n_examples or base in forced_examples):
            example_curves[base] = curves_for_example

    df = pd.DataFrame(rows)
    per_image_csv = os.path.join(out_root, "per_image_method_metrics.csv")
    df.to_csv(per_image_csv, index=False)
    print("Per-image metrics saved:", per_image_csv)

    if df.empty:
        pd.DataFrame().to_csv(os.path.join(out_root, "pivot_metrics.csv"))
        print("Evaluation complete. No Grad-CAM rows found.")
        return

    agg_dir = os.path.join(out_root, "aggregates")
    os.makedirs(agg_dir, exist_ok=True)

    metrics_to_plot = ['deletion_auc', 'insertion_auc', 'fraction_below_0.5', 'iou_mean', 'pointing_acc']
    for metric in metrics_to_plot:
        if metric not in df.columns or df[metric].dropna().size == 0:
            continue

        sum_df, pvals = aggregate_and_test(
            df[['image', 'method', metric]].dropna(),
            metric_col=metric,
            group_col='method',
            baseline=baseline_method
        )
        sum_csv = os.path.join(agg_dir, f"summary_{metric}.csv")
        sum_df.to_csv(sum_csv, index=False)

        pv_path = os.path.join(agg_dir, f"pvals_{metric}.json")
        with open(pv_path, "w") as f:
            json.dump(pvals, f, indent=2)

        plot_path = os.path.join(agg_dir, f"boxplot_{metric}.png")
        plot_boxplots(df[['image', 'method', metric]].dropna(), metric, plot_path, title=f"{metric} by method")
        print(f"Saved aggregate for {metric}: {sum_csv} and {plot_path}")

    examples_dir = os.path.join(out_root, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    for base, cmap in example_curves.items():
        curves_del = {m: (v[0], v[1]) for m, v in cmap.items()}
        plot_curves_example(
            curves_del,
            os.path.join(examples_dir, f"{base}_deletion_curves.png"),
            title=f"Deletion curves - {base}"
        )

        curves_ins = {m: (v[2], v[3]) for m, v in cmap.items()}
        plot_curves_example(
            curves_ins,
            os.path.join(examples_dir, f"{base}_insertion_curves.png"),
            title=f"Insertion curves - {base}"
        )

        methods_present = [
            m for m in methods
            if os.path.exists(os.path.join(results_root, "gradcam", m, f"{base}_overlay.png"))
        ]
        if methods_present:
            cols = len(methods_present)
            fig, axs = plt.subplots(1, cols, figsize=(3 * cols, 3))
            if cols == 1:
                axs = [axs]
            for i, m in enumerate(methods_present):
                p = os.path.join(results_root, "gradcam", m, f"{base}_overlay.png")
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) if os.path.exists(p) else np.zeros((256, 256, 3), dtype=np.uint8)
                axs[i].imshow(img)
                axs[i].axis('off')
                axs[i].set_title(m)
            plt.suptitle(f"Overlays - {base}")
            plt.tight_layout()
            plt.savefig(os.path.join(examples_dir, f"{base}_overlay_grid.png"), dpi=200)
            plt.close()

    pivot = df.pivot_table(
        index='image',
        columns='method',
        values=['deletion_auc', 'insertion_auc', 'fraction_below_0.5', 'iou_mean', 'pointing_acc']
    )
    pivot_csv = os.path.join(out_root, "pivot_metrics.csv")
    pivot.to_csv(pivot_csv)
    print("Pivot saved:", pivot_csv)

    print("Evaluation complete. Check folder:", out_root)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--network', default=None)
    parser.add_argument('--gt_path', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--baseline_method', default='baseline')
    args = parser.parse_args()

    evaluate_all(
        results_root=args.results_root,
        image_folder=args.image_folder,
        network=args.network,
        gt_path=args.gt_path,
        steps=args.steps,
        batch=args.batch,
        baseline_method=args.baseline_method,
        device=args.device
    )