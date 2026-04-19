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
    cand_paths = [

        os.path.join(results_root, "rise", method, f"{base}_raw.npy"),
        os.path.join(results_root, "rise", method, f"{base}_rise.npy"),
        os.path.join(results_root, "rise", method, f"{base}.npy"),
        os.path.join(results_root, "rise", method, f"{base}_sal.npy"),
        os.path.join(results_root, "rise", method, f"{base}_heatmap.npy"),

        os.path.join(results_root, "rise", base, f"{method}_raw.npy"),
        os.path.join(results_root, "rise", base, f"{method}_rise.npy"),

        os.path.join(results_root, "rise", f"{base}_rise.npy"),
        os.path.join(results_root, "rise", f"{base}_raw.npy"),
        os.path.join(results_root, "rise", f"{base}.npy"),

        os.path.join(results_root, f"{base}_rise.npy"),
        os.path.join(results_root, f"{base}.npy"),
    ]

    for p in cand_paths:
        if os.path.exists(p):

            if p.lower().endswith(".npy"):
                try:
                    return np.load(p)
                except Exception as e:
                    print(f"evaluation_rise.load_heatmap: Found {p} but failed to np.load(): {e}")
                    return None
            else:

                continue

    overlay_candidates = [
        os.path.join(results_root, "rise", method, f"{base}_rise_overlay.png"),
        os.path.join(results_root, "rise", method, f"{base}_overlay.png"),
        os.path.join(results_root, "rise", f"{base}_rise_overlay.png"),
        os.path.join(results_root, "rise", f"{base}_overlay.png"),
    ]
    for p in overlay_candidates:
        if os.path.exists(p):
            print(f"evaluation_rise.load_heatmap: found overlay image but no .npy (overlay='{p}').")
            return None

    return None


def list_basenames_from_folder(image_folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith(exts)])


def read_gt_bboxes(gt_path):
    if gt_path is None or not os.path.exists(gt_path):
        return {}
    if gt_path.lower().endswith(".json"):
        with open(gt_path, "r") as f:
            j = json.load(f)
        for k, v in j.items():
            j[k] = [[int(round(c)) for c in bb] for bb in v]
        return j
    else:
        df = pd.read_csv(gt_path, header=None)
        gt = {}
        for _, row in df.iterrows():
            base = str(row[0])
            bb = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
            gt.setdefault(base, []).append(bb)
        return gt


def load_image_tensor(image_folder, base, device):
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = os.path.join(image_folder, base + ext)
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            return transform(img).unsqueeze(0).to(device)
    raise FileNotFoundError(f"Image {base} not found in {image_folder}")


def compose_image_with_mask(orig_tensor, mask, mode='blur'):
    device = orig_tensor.device
    inp = orig_tensor.detach().cpu().numpy()[0]
    if inp.min() < 0 or inp.max() > 1:
        vis = inp * 0.5 + 0.5
    else:
        vis = inp.copy()
    vis = np.transpose(vis, (1, 2, 0))
    mask3 = np.stack([mask] * 3, axis=2)
    if mode == 'zero':
        composed = vis * mask3
    elif mode == 'blur':
        blurred = cv2.GaussianBlur((vis * 255).astype(np.uint8), (31, 31), 0).astype(np.float32) / 255.0
        composed = vis * mask3 + blurred * (1 - mask3)
    elif mode == 'mean':
        mean_color = vis.mean(axis=(0, 1), keepdims=True)
        composed = vis * mask3 + mean_color * (1 - mask3)
    t = torch.from_numpy(np.transpose(composed, (2, 0, 1))).unsqueeze(0).to(device).float()
    if inp.min() < 0 or inp.max() > 1:
        t = (t - 0.5) / 0.5
    return t


def compute_order_from_map(sal_map):
    return np.argsort(-sal_map.flatten())


def batched_forward(model, batch_t, device):
    with torch.no_grad():
        out = model(batch_t.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()


def deletion_insertion_auc(model, orig_tensor, sal_map, class_idx, steps=50, mode='deletion', baseline_mode='blur',
                           device='cuda', batch_size=16):
    H, W = sal_map.shape
    total_pix = H * W
    order = compute_order_from_map(sal_map)
    step = int(np.ceil(total_pix / steps))
    fractions, scores = [], []

    for k in [min(s * step, total_pix) for s in range(0, steps + 1)]:
        if mode == 'deletion':
            mask_flat = np.ones(total_pix, dtype=np.float32)
            mask_flat[order[:k]] = 0
            mask = mask_flat.reshape(H, W)
            inp = compose_image_with_mask(orig_tensor, mask, mode=baseline_mode)
        else:
            mask_flat = np.zeros(total_pix, dtype=np.float32)
            mask_flat[order[:k]] = 1
            mask = mask_flat.reshape(H, W)
            inp = compose_image_with_mask(orig_tensor, mask, mode='mean')
        p = batched_forward(model, inp, device)[0, class_idx]
        scores.append(p)
        fractions.append(k / float(total_pix))
    return float(np.trapz(scores, fractions)), np.array(scores), np.array(fractions)


def sparsity_metrics(sal_map, threshold=0.5):
    H, W = sal_map.shape
    mask_below = (sal_map < threshold).astype(np.uint8)
    fraction_below = mask_below.sum() / float(H * W)
    labeled = label(1 - mask_below)
    areas = [r.area for r in regionprops(labeled)] if labeled.max() > 0 else []
    avg_area = (np.mean(areas) / (H * W)) if areas else 0.0
    return fraction_below, int(labeled.max()), avg_area


def iou_with_bbox(sal_map, bbox, threshold=0.5):
    H, W = sal_map.shape
    mask = (sal_map > threshold).astype(np.uint8)
    xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
    bbox_mask = np.zeros_like(mask)
    bbox_mask[ymin:ymax + 1, xmin:xmax + 1] = 1
    inter = (mask & bbox_mask).sum()
    union = (mask | bbox_mask).sum()
    return 0.0 if union == 0 else inter / union


def pointing_game(sal_map, bbox):
    H, W = sal_map.shape
    y, x = divmod(int(np.argmax(sal_map)), W)
    xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
    return xmin <= x <= xmax and ymin <= y <= ymax


def aggregate_and_test(df, metric_col, group_col='method', baseline='baseline'):
    methods = df[group_col].unique().tolist()
    summary, pvals = [], {}
    for m in methods:
        arr = df[df[group_col] == m][metric_col].dropna().values
        summary.append({'method': m, 'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'n': int(arr.size)})
        if m != baseline:
            merged = pd.merge(df[df[group_col] == baseline][['image', metric_col]],
                              df[df[group_col] == m][['image', metric_col]],
                              on='image', suffixes=('_base', '_m'))
            a = merged[metric_col + '_base'].values
            b = merged[metric_col + '_m'].values
            try:
                stat, p = stats.wilcoxon(a, b)
            except:
                stat, p = stats.ttest_rel(a, b)
            pvals[m] = float(p)
    return pd.DataFrame(summary), pvals


def plot_boxplots(df, metric_col, out_path, title=None):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='method', y=metric_col, data=df, order=sorted(df['method'].unique()))
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


def evaluate_all(results_root, image_folder, network=None, gt_path=None,
                 methods=None, steps=50, batch=16, n_examples=6, baseline_method='baseline', device=None):
    if methods is None:
        methods = ['baseline', 'sift']
    out_root = os.path.join(results_root, "evaluation_rise")
    os.makedirs(out_root, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    basenames = list_basenames_from_folder(image_folder)
    gt = read_gt_bboxes(gt_path) if gt_path else {}

    model = None
    if network and has_utils:
        D_raw = load_discriminator(network)

        class DiscriminatorTwoClassWrapper(torch.nn.Module):
            def __init__(self, d):
                super().__init__();
                self.discriminator = d

            def forward(self, x):
                logit = self.discriminator(x, None)
                return torch.nn.functional.softmax(torch.cat([-logit, logit], dim=1), dim=1)

        model = DiscriminatorTwoClassWrapper(D_raw).to(device).eval()

    rows = []
    forced_examples = {"seed0151", "seed0152"}
    example_curves = {}

    for base in tqdm(basenames, desc="Evaluating images"):
        img_tensor = load_image_tensor(image_folder, base, device) if model else None
        curves_for_example = {}

        for m in methods:
            heat = load_heatmap(results_root, m, base)
            if heat is None: continue
            heat2 = heat.mean(axis=2) if heat.ndim == 3 else heat
            heat2 = (heat2 - heat2.min()) / (heat2.max() + 1e-8) if heat2.max() > 0 else heat2

            frac_above, n_comp, avg_area = sparsity_metrics(heat2)
            del_auc, ins_auc = np.nan, np.nan

            if model is not None and img_tensor is not None:
                probs = model(img_tensor);
                class_idx = int(probs.argmax(dim=1)[0].item())
                del_auc, del_scores, del_fracs = deletion_insertion_auc(model, img_tensor, heat2, class_idx,
                                                                        steps=steps, mode='deletion',
                                                                        baseline_mode='blur', device=device,
                                                                        batch_size=batch)
                ins_auc, ins_scores, ins_fracs = deletion_insertion_auc(model, img_tensor, heat2, class_idx,
                                                                        steps=steps, mode='insertion',
                                                                        baseline_mode='blur', device=device,
                                                                        batch_size=batch)
                curves_for_example[m] = (del_scores, del_fracs, ins_scores, ins_fracs)

            iou_mean, pointing_acc = np.nan, np.nan
            if base in gt:
                ious, points = [], []
                for bb in gt[base]:
                    ious.append(iou_with_bbox(heat2, bb))
                    points.append(1.0 if pointing_game(heat2, bb) else 0.0)
                iou_mean, pointing_acc = np.nanmean(ious), np.nanmean(points)

            rows.append({
                'image': base, 'method': m,
                'deletion_auc': del_auc, 'insertion_auc': ins_auc,
                'fraction_below_0.5': frac_above, 'n_components': n_comp, 'avg_component_area': avg_area,
                'iou_mean': iou_mean, 'pointing_acc': pointing_acc
            })

        if curves_for_example and (len(example_curves) < n_examples or base in forced_examples):
            example_curves[base] = curves_for_example

    # ---- Save Data ----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "per_image_method_metrics.csv"), index=False)

    agg_dir = os.path.join(out_root, "aggregates")
    os.makedirs(agg_dir, exist_ok=True)
    metrics_to_plot = ['deletion_auc', 'insertion_auc', 'fraction_below_0.5', 'iou_mean', 'pointing_acc']
    for metric in metrics_to_plot:
        if metric not in df.columns or df[metric].dropna().size == 0: continue
        sum_df, pvals = aggregate_and_test(df[['image', 'method', metric]].dropna(), metric_col=metric,
                                           baseline=baseline_method)
        sum_df.to_csv(os.path.join(agg_dir, f"summary_{metric}.csv"), index=False)
        with open(os.path.join(agg_dir, f"pvals_{metric}.json"), "w") as f:
            json.dump(pvals, f, indent=2)
        plot_boxplots(df[['image', 'method', metric]].dropna(), metric, os.path.join(agg_dir, f"boxplot_{metric}.png"))

    examples_dir = os.path.join(out_root, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    for base, cmap in example_curves.items():
        curves_del = {m: (v[0], v[1]) for m, v in cmap.items()}
        curves_ins = {m: (v[2], v[3]) for m, v in cmap.items()}
        plot_curves_example(curves_del, os.path.join(examples_dir, f"{base}_deletion_curves.png"),
                            title=f"Deletion - {base}")
        plot_curves_example(curves_ins, os.path.join(examples_dir, f"{base}_insertion_curves.png"),
                            title=f"Insertion - {base}")

    pivot = df.pivot_table(index='image', columns='method',
                           values=['deletion_auc', 'insertion_auc', 'fraction_below_0.5', 'iou_mean', 'pointing_acc'])
    pivot.to_csv(os.path.join(out_root, "pivot_metrics.csv"), index=False)

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
