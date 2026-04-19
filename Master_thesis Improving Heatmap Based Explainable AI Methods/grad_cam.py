
import os
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer {layer_name} nicht gefunden. Verfügbare Layer-Names (Ausschnitt): " +
                     ", ".join([n for n, _ in list(model.named_modules())[:200]]))

def _img_tensor_to_display_np(inp_tensor):
    
    inp = inp_tensor.detach().cpu()[0].clone()
    arr = inp.numpy()
    arr = np.transpose(arr, (1,2,0))
    
    arr = (arr * 0.5) + 0.5
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)

def _kaze_blob_from_np(img_np, n_keypoints=300, blob_radius=12):
    
    img255 = (img_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(img255, cv2.COLOR_RGB2GRAY)
    kaze = cv2.KAZE_create()
    kps = kaze.detect(gray, None)
    if not kps:
        return np.zeros((gray.shape[0], gray.shape[1]), dtype=np.float32)
    kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:n_keypoints]
    blob = np.zeros_like(gray, dtype=np.float32)
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(blob, (x, y), max(1, blob_radius), 1.0, -1)
    sigma = max(1.0, blob_radius/2.0)
    blob = cv2.GaussianBlur(blob, (0,0), sigmaX=sigma, sigmaY=sigma)
    if blob.max() > 0:
        blob = blob / (blob.max() + 1e-8)
    return blob.astype(np.float32)

# -------------------------
# Guided Backprop
# -------------------------
class GuidedBackprop:
   
    def __init__(self, model):
        self.model = model.eval()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                try:
                    h = module.register_full_backward_hook(lambda mod, grad_in, grad_out: tuple(
                        (g.clamp(min=0.0) if g is not None else None) for g in grad_in))
                except Exception:
                    h = module.register_backward_hook(lambda mod, grad_in, grad_out: tuple(
                        (g.clamp(min=0.0) if g is not None else None) for g in grad_in))
                self.hooks.append(h)

    def generate_gradients(self, input_tensor, target_index=None):
        inp = input_tensor.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(inp)
        if target_index is None:
            target_index = int(out.argmax(dim=1)[0].item())
        loss = out[0, target_index]
        loss.backward(retain_graph=True)
        grad = inp.grad[0].detach().cpu().numpy() 
        grad = np.transpose(grad, (1,2,0))
        grad = np.abs(grad)
        grad = grad - grad.min()
        if grad.max() > 0:
            grad = grad / (grad.max() + 1e-8)
        return grad.astype(np.float32)

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass

# -------------------------
# Grad-CAM Variants
# -------------------------
def baseline_gradcam(model, input_tensor, target_layer_name, class_idx, upsample_size=(1024,1024), device='cuda'):
   
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    try:
        target_module = get_layer_by_name(model.discriminator, target_layer_name)
    except Exception:
        target_module = get_layer_by_name(model, target_layer_name)

    targets = [ClassifierOutputTarget(class_idx)]

    try:
        cam = GradCAM(model=model, target_layers=[target_module])
    except Exception as e:
        
        raise RuntimeError(f"GradCAM Init Fehler: {e}")

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    W, H = upsample_size[0], upsample_size[1]
    if grayscale_cam.shape != (H, W):
        grayscale_cam = cv2.resize(grayscale_cam, (W, H), interpolation=cv2.INTER_CUBIC)

    grayscale_cam = grayscale_cam - grayscale_cam.min()
    if grayscale_cam.max() > 0:
        grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
    return grayscale_cam.astype(np.float32)

def multilayer_gradcam(model, input_tensor, target_layer_names, class_idx,
                       upsample_size=(1024,1024), layer_weights=None, device='cuda'):
    
    cams = []
    for lname in target_layer_names:
        c = baseline_gradcam(model, input_tensor, lname, class_idx, upsample_size=upsample_size, device=device)
        cams.append(c)
    cams = np.stack(cams, axis=0) 
    if layer_weights is None:
        final = np.mean(cams, axis=0)
    else:
        w = np.array(layer_weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        final = np.tensordot(w, cams, axes=(0,0))
    final = final - final.min()
    if final.max() > 0:
        final = final / (final.max() + 1e-8)
    return final.astype(np.float32)

def guided_gradcam(model, input_tensor, target_layer_name, class_idx, upsample_size=(1024,1024), device='cuda'):
   
    cam_map = baseline_gradcam(model, input_tensor, target_layer_name, class_idx, upsample_size=upsample_size, device=device)
    gb = GuidedBackprop(model)
    try:
        gb_map = gb.generate_gradients(input_tensor, target_index=class_idx)  # HxWx3
    finally:
        gb.remove_hooks()
    cam_3 = np.expand_dims(cam_map, axis=2)
    guided = gb_map * cam_3
    guided = guided - guided.min()
    if guided.max() > 0:
        guided = guided / (guided.max() + 1e-8)
    return guided.astype(np.float32)

def feature_fusion_gradcam(model, input_tensor, target_layer_names, class_idx,
                           upsample_size=(1024,1024),
                           layer_weights=None,
                           kaze_params=None,
                           device='cuda'):
  
    cam = multilayer_gradcam(model, input_tensor, target_layer_names, class_idx,
                             upsample_size=upsample_size, layer_weights=layer_weights, device=device)
    display_img = _img_tensor_to_display_np(input_tensor)
    kparams = kaze_params or {'n_keypoints':300, 'blob_radius':12}
    blob = _kaze_blob_from_np(display_img, n_keypoints=kparams.get('n_keypoints',300),
                              blob_radius=kparams.get('blob_radius',12))
    H = upsample_size[1]; W = upsample_size[0]
    if blob.shape != (H, W):
        blob = cv2.resize(blob, (W, H), interpolation=cv2.INTER_CUBIC)
        if blob.max() > 0:
            blob = blob / (blob.max() + 1e-8)
    fused = cam * (0.6 + 0.4 * blob)
    fused = fused - fused.min()
    if fused.max() > 0:
        fused = fused / (fused.max() + 1e-8)
    return fused.astype(np.float32)

def combined_supermap(model, input_tensor, baseline_layer, multilayer_names, class_idx,
                      upsample_size=(1024,1024), layer_weights=None, kaze_params=None, device='cuda'):
    
    baseline = baseline_gradcam(model, input_tensor, baseline_layer, class_idx, upsample_size=upsample_size, device=device)
    multilayer = multilayer_gradcam(model, input_tensor, multilayer_names, class_idx,
                                   upsample_size=upsample_size, layer_weights=layer_weights, device=device)
    guided_rgb = guided_gradcam(model, input_tensor, baseline_layer, class_idx, upsample_size=upsample_size, device=device)
    guided_gray = guided_rgb.mean(axis=2)
    feature = feature_fusion_gradcam(model, input_tensor, multilayer_names, class_idx,
                                     upsample_size=upsample_size, layer_weights=layer_weights, kaze_params=kaze_params, device=device)
    # Gewichtete Kombination
    w_baseline = 0.25
    w_multi = 0.25
    w_guided = 0.25
    w_feat = 0.25
    combined = (w_baseline * baseline) + (w_multi * multilayer) + (w_guided * guided_gray) + (w_feat * feature)
    combined = combined - combined.min()
    if combined.max() > 0:
        combined = combined / (combined.max() + 1e-8)
    return combined.astype(np.float32)


def save_overlay_and_raw(save_base_dir, base_name, display_img, map_arr, overlay_fn=show_cam_on_image):
   
    os.makedirs(save_base_dir, exist_ok=True)
    raw_path = os.path.join(save_base_dir, base_name + "_raw.npy")
    np.save(raw_path, map_arr)
    
    if map_arr.ndim == 3 and map_arr.shape[2] == 3:
        gray = map_arr.mean(axis=2)
        overlay = overlay_fn(display_img, gray, use_rgb=True)
    else:
        overlay = overlay_fn(display_img, map_arr, use_rgb=True)
    overlay_path = os.path.join(save_base_dir, base_name + "_overlay.png")
    
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    heatmap_path = os.path.join(save_base_dir, base_name + "_heatmap.png")
    hm = (map_arr if map_arr.ndim==2 else map_arr.mean(axis=2))
    hm_img = (255 * np.clip(hm, 0.0, 1.0)).astype(np.uint8)
    cv2.imwrite(heatmap_path, hm_img)
    return raw_path, overlay_path, heatmap_path
