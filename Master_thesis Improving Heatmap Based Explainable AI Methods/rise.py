import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def _normalize_map(m):
    m = m.astype(np.float32)
    m -= m.min()
    if m.max() > 0:
        m /= m.max()
    return m


def _apply_jet_colormap_uint8(map_norm):
    m = np.clip(map_norm, 0.0, 1.0)
    jet_bgr = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    return jet_rgb


def _prepare_display_img(orig_img):
    if torch.is_tensor(orig_img):
        arr = orig_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    elif isinstance(orig_img, np.ndarray):
        arr = orig_img
    else:
        raise TypeError(f"Unsupported image type for overlay: {type(orig_img)}")

    arr = arr.astype(np.float32)

    if arr.min() < 0.0:
        arr = (arr * 0.5) + 0.5

    if arr.max() > 1.0:
        arr = np.clip(arr / 255.0, 0.0, 1.0)

    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _blend_overlay_with_display(display_img, jet_rgb_u8, map_norm, alpha=0.45):
    heat = jet_rgb_u8.astype(np.float32) / 255.0
    map_exp = np.expand_dims(np.clip(map_norm, 0.0, 1.0), axis=2)
    alpha_map = alpha * map_exp
    overlay = display_img * (1.0 - alpha_map) + heat * alpha_map
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def _save_overlay(map2d, orig_img, out_folder, base_name, base_alpha=0.45):
    os.makedirs(out_folder, exist_ok=True)

    display_img = _prepare_display_img(orig_img)

    map_norm = map2d.copy().astype(np.float32)
    if map_norm.min() < 0 or map_norm.max() > 1:
        denom = map_norm.max() - map_norm.min() + 1e-10
        map_norm = (map_norm - map_norm.min()) / denom

    H, W = display_img.shape[0], display_img.shape[1]
    if map_norm.shape != (H, W):
        map_img = Image.fromarray((map_norm * 255).astype(np.uint8))
        map_img = map_img.resize((W, H), resample=Image.BILINEAR)
        map_norm = np.array(map_img).astype(np.float32) / 255.0

    jet_rgb_u8 = _apply_jet_colormap_uint8(map_norm)
    overlay = _blend_overlay_with_display(display_img, jet_rgb_u8, map_norm, alpha=base_alpha)

    overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_folder, f"{base_name}_rise_overlay.png"), overlay_bgr)


def _save_map_and_np(map2d, out_folder, base_name, orig_img=None):
    os.makedirs(out_folder, exist_ok=True)
    map_norm = _normalize_map(map2d)

    np.save(os.path.join(out_folder, f"{base_name}_rise.npy"), map_norm)

    jet_rgb = _apply_jet_colormap_uint8(map_norm)
    jet_bgr = cv2.cvtColor(jet_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_folder, f"{base_name}_rise.png"), jet_bgr)

    if orig_img is not None:
        _save_overlay(map_norm, orig_img, out_folder, base_name, base_alpha=0.45)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"_to_numpy: unsupported type {type(x)}")


def _upsample_masks(masks, out_h, out_w, mode='bilinear'):
    if isinstance(masks, torch.Tensor) and masks.dim() == 3:
        masks = masks.unsqueeze(1)
    return F.interpolate(masks, size=(out_h, out_w), mode=mode, align_corners=False)


def _batched_forward_numpy_preds(model, imgs, device, batch_size=64):
    model = model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            batch = imgs[i:i + batch_size].to(device)
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds.append(out.detach().cpu().numpy())

    if len(preds) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    return np.concatenate(preds, axis=0)


def _apply_mask_with_same_normalization(
        img_tensor,
        masks_batch,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5)
):
    device = img_tensor.device
    mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)

    img_unnorm = img_tensor * std + mean
    b = masks_batch.shape[0]
    imgs_b_unnorm = img_unnorm.repeat(b, 1, 1, 1) * masks_batch.repeat(1, 3, 1, 1)
    imgs_b = (imgs_b_unnorm - mean) / std
    return imgs_b


def generate_random_masks(n_masks, mask_size, img_h, img_w, p=0.5, device='cpu', multi_scale=False):
    if multi_scale:
        sizes = [max(2, mask_size // 2), mask_size, mask_size * 2]
    else:
        sizes = [mask_size]

    target = max(sizes)
    masks_list = []

    for s in sizes:
        n_per = int(math.ceil(n_masks / len(sizes)))
        rand = torch.rand((n_per, 1, s, s), device='cpu')
        binm = (rand < p).float()
        if s != target:
            binm = F.interpolate(binm, size=(target, target), mode='nearest')
        masks_list.append(binm)

    masks = torch.cat(masks_list, dim=0)[:n_masks]
    return masks.to(device)


def _apply_masks_and_get_scores_chunk(
        model,
        img_tensor,
        masks_hr_torch,
        class_idx,
        device,
        batch_size=64,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5)
):
    N = masks_hr_torch.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.float32)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    img = img_tensor.to(device)
    scores_list = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(0, N, batch_size):
            m_batch = masks_hr_torch[i:i + batch_size].to(device)
            imgs_b = _apply_mask_with_same_normalization(
                img,
                m_batch,
                norm_mean=norm_mean,
                norm_std=norm_std
            )
            preds = _batched_forward_numpy_preds(model, imgs_b, device=device, batch_size=batch_size)
            scores_batch = preds[:, class_idx]
            scores_list.append(scores_batch)

    if len(scores_list) == 0:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(scores_list, axis=0)


def baseline_rise(
        model,
        input_tensor,
        class_idx,
        N=250,
        s=16,
        p1=0.5,
        batch_size=32,
        device=None,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5)
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _, C, H, W = input_tensor.shape

    masks = np.random.rand(N, s, s) < p1
    masks = masks.astype(np.float32)
    masks = torch.tensor(masks).unsqueeze(1).to(device)
    masks_hr = _upsample_masks(masks, H, W, mode='bilinear')
    mask_np = _to_numpy(masks_hr.squeeze(1))

    scores = _apply_masks_and_get_scores_chunk(
        model,
        input_tensor,
        masks_hr,
        class_idx,
        device=device,
        batch_size=batch_size,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    if scores.shape[0] != mask_np.shape[0]:
        N_eff = scores.shape[0]
        mask_np = mask_np[:N_eff]
        scores = scores[:N_eff]

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap_acc = (scores_norm[:, None, None] * mask_np).sum(axis=0)
    denom = mask_np.sum(axis=0) + 1e-8
    saliency = heatmap_acc / denom / p1
    return saliency.astype(np.float32)


def focus_rise(
        model,
        input_tensor,
        class_idx,
        n_masks=2000,
        mask_size=8,
        p=0.5,
        batch_size=64,
        device=None,
        gauss_sigma=0.4,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5)
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _, C, H, W = input_tensor.shape

    yy = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
    xx = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
    gauss = torch.exp(-((xx ** 2 + yy ** 2) / (gauss_sigma ** 2)))
    gauss_np = _to_numpy(gauss)

    masks = generate_random_masks(n_masks, mask_size, H, W, p=p, device='cpu', multi_scale=True)
    masks_hr = _upsample_masks(masks, H, W, mode='bilinear').to(device)
    mask_np = _to_numpy(masks_hr.squeeze(1))

    scores = _apply_masks_and_get_scores_chunk(
        model,
        input_tensor,
        masks_hr,
        class_idx,
        device=device,
        batch_size=batch_size,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    if scores.shape[0] != mask_np.shape[0]:
        n_eff = min(scores.shape[0], mask_np.shape[0])
        scores = scores[:n_eff]
        mask_np = mask_np[:n_eff]

    overlap = (mask_np * gauss_np[None, :, :]).reshape(mask_np.shape[0], -1).sum(axis=1)
    overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min() + 1e-8)

    heatmap_acc = (scores[:, None, None] * overlap[:, None, None] * mask_np).sum(axis=0)
    denom = mask_np.sum(axis=0) * (overlap.mean() + 1e-8)
    heatmap = heatmap_acc / denom
    return heatmap.astype(np.float32)


def contrast_rise(
        model,
        input_tensor,
        class_idx,
        n_masks=2000,
        mask_size=8,
        p=0.5,
        batch_size=64,
        device=None,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5)
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _, C, H, W = input_tensor.shape
    img_np = _to_numpy(input_tensor.detach().cpu())[0].transpose(1, 2, 0)

    masks = generate_random_masks(n_masks, mask_size, H, W, p=p, device='cpu')
    masks_hr = _upsample_masks(masks, H, W, mode='bilinear').to(device)
    mask_np = _to_numpy(masks_hr.squeeze(1))

    masked = mask_np[:, :, :, None] * img_np[None, :, :, :]
    diffs = np.abs(img_np[None, :, :, :] - masked).mean(axis=(1, 2, 3))
    diffs_norm = (diffs - diffs.min()) / (diffs.max() - diffs.min() + 1e-8)

    scores = _apply_masks_and_get_scores_chunk(
        model,
        input_tensor,
        masks_hr,
        class_idx,
        device=device,
        batch_size=batch_size,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    if scores.shape[0] != mask_np.shape[0]:
        n_eff = min(scores.shape[0], mask_np.shape[0])
        scores = scores[:n_eff]
        mask_np = mask_np[:n_eff]
        diffs_norm = diffs_norm[:n_eff]

    heatmap_acc = (scores[:, None, None] * diffs_norm[:, None, None] * mask_np).sum(axis=0)
    denom = mask_np.sum(axis=0) * (diffs_norm.mean() + 1e-8)
    heatmap = heatmap_acc / denom
    return heatmap.astype(np.float32)


def combined_supermap_from_maps(maps_dict):
    if len(maps_dict) == 0:
        raise ValueError("combined_supermap_from_maps received an empty maps_dict")

    names = list(maps_dict.keys())
    normed = {}

    for k in names:
        arr = maps_dict[k].astype(np.float32)
        arr = _normalize_map(arr)
        normed[k] = arr

    baseline = normed.get('baseline', np.zeros_like(next(iter(normed.values()))))
    exts = [normed[k] for k in normed if k != 'baseline']

    w_base, w_ext = 1.0, 1.2
    combined = w_base * baseline.copy()

    if len(exts) > 0:
        combined += w_ext * sum(exts) / len(exts)

    stack = np.stack(list(normed.values()), axis=0)
    consensus = (stack > 0.6).sum(axis=0) / float(stack.shape[0])
    combined = combined * (1.0 + 0.5 * consensus)
    combined = np.sqrt(combined)
    combined = _normalize_map(combined)
    return combined


def extract_sift_density_map(orig_img, H, W):
    if orig_img is None:
        raise ValueError("extract_sift_density_map: orig_img is None")

    if torch.is_tensor(orig_img):
        img = orig_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        img = ((img * 0.5) + 0.5) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif isinstance(orig_img, np.ndarray):
        img = orig_img.copy()
    else:
        raise TypeError(f"extract_sift_density_map: unsupported orig_img type {type(orig_img)}")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"extract_sift_density_map: expected RGB image with shape (H,W,3), got {img.shape}")

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=800,
        contrastThreshold=0.02,
        edgeThreshold=8,
        sigma=1.2
    )
    keypoints = sift.detect(gray, None)

    density = np.zeros((H, W), dtype=np.float32)

    if keypoints is None:
        keypoints = []

    # robust normalization for SIFT responses
    responses = np.array(
        [max(kp.response, 1e-6) for kp in keypoints],
        dtype=np.float32
    ) if len(keypoints) > 0 else np.array([1.0], dtype=np.float32)

    resp_ref = np.percentile(responses, 90) + 1e-8

    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= y < H and 0 <= x < W:
            weight = min(float(max(kp.response, 1e-6) / resp_ref), 1.0)
            radius = max(1, int(round(0.35 * kp.size)))
            cv2.circle(density, (x, y), radius, weight, -1)

    density = cv2.GaussianBlur(density, (7, 7), 0)
    density = _normalize_map(density)

    # remove very weak residual regions
    density[density < 0.05] = 0.0

    density = _normalize_map(density)
    return density

def generate_sift_guided_masks(n_masks, mask_size, H, W, density_map, device='cpu'):
    masks = []

    density_small = cv2.resize(density_map, (mask_size, mask_size), interpolation=cv2.INTER_LINEAR)
    density_small = density_small.astype(np.float32)

    if density_small.max() > 0:
        density_small = density_small / (density_small.max() + 1e-8)

    # emphasize strong regions, but keep enough flexibility
    density_small = density_small ** 1.8

    # identify stronger structural regions
    if np.any(density_small > 0):
        thr = np.percentile(density_small[density_small > 0], 65)
    else:
        thr = 0.0

    strong = (density_small >= thr).astype(np.float32)

    # probability map: guided, but not too rigid
    prob = np.clip(0.03 + 0.35 * strong + 0.30 * density_small, 0.0, 0.70)

    for _ in range(n_masks):
        random_mask = np.random.rand(mask_size, mask_size)
        binary_mask = (random_mask < prob).astype(np.float32)

        # resize to image size
        mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_LINEAR)

        # soften mask borders to reduce blockiness
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask = np.clip(mask, 0.0, 1.0)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        masks.append(mask)

    masks = torch.cat(masks, dim=0)
    return masks.to(device)

def sift_only_rise(model, input_tensor, class_idx, orig_img,
                   n_masks=200, mask_size=8, batch_size=32, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _, C, H, W = input_tensor.shape

    # pure SIFT structural prior only
    density_map = extract_sift_density_map(orig_img, H, W)

    masks_hr = generate_sift_guided_masks(
        n_masks, mask_size, H, W, density_map, device=device
    )

    mask_np = _to_numpy(masks_hr.squeeze(1))

    # overlap between each mask and SIFT prior
    overlap = (mask_np * density_map[None, :, :]).reshape(mask_np.shape[0], -1).sum(axis=1)
    overlap = overlap - overlap.min()
    if overlap.max() > 0:
        overlap = overlap / (overlap.max() + 1e-8)

    # softer weighting than before to avoid over-selecting only a few masks
    overlap = overlap ** 1.6
    overlap[overlap < 0.05] = 0.0

    scores = _apply_masks_and_get_scores_chunk(
        model,
        input_tensor,
        masks_hr,
        class_idx,
        device=device,
        batch_size=batch_size
    )

    if scores.shape[0] != mask_np.shape[0]:
        n_eff = min(scores.shape[0], mask_np.shape[0])
        scores = scores[:n_eff]
        mask_np = mask_np[:n_eff]
        overlap = overlap[:n_eff]

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # soften score dominance
    scores_norm = np.sqrt(scores_norm)

    heatmap_acc = (scores_norm[:, None, None] * overlap[:, None, None] * mask_np).sum(axis=0)
    denom = (overlap[:, None, None] * mask_np).sum(axis=0) + 1e-8
    saliency = heatmap_acc / denom

    # slightly compress peaks for smoother and more readable maps
    saliency = np.power(np.clip(saliency, 0.0, None), 0.85)

    # final smoothing
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    return _normalize_map(saliency)

def generate_rise(model, input_tensor, class_idx,
                  result_root=None, base_name=None, orig_img=None,
                  variants=None,
                  n_masks=4000, mask_size=8, p=0.5,
                  batch_size=64, chunk_size=256, device=None,
                  norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5)):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if variants is None:
        variants = ['baseline']

    results_maps = {}

    if 'baseline' in variants:
        try:
            m = baseline_rise(
                model, input_tensor, class_idx,
                N=250, s=16, p1=0.5,
                batch_size=batch_size, device=device,
                norm_mean=norm_mean, norm_std=norm_std
            )
            results_maps['baseline'] = _normalize_map(m)
        except Exception as e:
            print("Warning: baseline_rise failed:", e)

    if 'focus' in variants:
        try:
            m = focus_rise(
                model, input_tensor, class_idx,
                n_masks=n_masks, mask_size=mask_size, p=p,
                batch_size=batch_size, device=device,
                norm_mean=norm_mean, norm_std=norm_std
            )
            results_maps['focus'] = _normalize_map(m)
        except Exception as e:
            print("Warning: focus_rise failed:", e)

    if 'contrast' in variants:
        try:
            m = contrast_rise(
                model, input_tensor, class_idx,
                n_masks=n_masks, mask_size=mask_size, p=p,
                batch_size=batch_size, device=device,
                norm_mean=norm_mean, norm_std=norm_std
            )
            results_maps['contrast'] = _normalize_map(m)
        except Exception as e:
            print("Warning: contrast_rise failed:", e)


    if 'sift_only' in variants:
        try:
            print("DEBUG: entered sift_only block")
            print("DEBUG: orig_img type:", type(orig_img))
            print("DEBUG: orig_img shape:", getattr(orig_img, "shape", None))
            print("DEBUG: input_tensor shape:", tuple(input_tensor.shape))

            m = sift_only_rise(
                model=model,
                input_tensor=input_tensor,
                class_idx=class_idx,
                orig_img=orig_img,
                n_masks=n_masks,
                mask_size=mask_size,
                batch_size=batch_size,
                device=device
            )

            print("DEBUG: sift_only_rise finished")
            print("DEBUG: sift_only map min/max:", np.min(m), np.max(m))

            results_maps['sift_only'] = _normalize_map(m)
            print("DEBUG: sift_only added to results_maps")

        except Exception as e:
            print("DEBUG: sift_only_rise failed:", repr(e))



    if 'combined' in variants:
        try:
            if len(results_maps) > 0:
                m = combined_supermap_from_maps(results_maps)
                results_maps['combined'] = _normalize_map(m)
            else:
                print("Warning: combined_supermap skipped because results_maps is empty")
        except Exception as e:
            print("Warning: combined_supermap failed:", e)

    # if 'baseline' in results_maps and 'grad' in results_maps:
    #     try:
    #         results_maps['baseline_refined_by_grad'] = _normalize_map(
    #             results_maps['baseline'] * results_maps['grad']
    #         )
    #     except Exception:
    #         pass

    if result_root is not None and base_name is not None:
        rise_root = os.path.join(result_root, "rise")
        for k, v in results_maps.items():
            print("DEBUG: saving", k)
            folder = os.path.join(rise_root, k)
            _save_map_and_np(v, folder, base_name, orig_img=orig_img)

    return results_maps
