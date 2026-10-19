"""Anomaly map between photos and drawings of the healthy object.

Same recipe as SplatPose / PAD: squared difference of the pixels plus squared differences of EfficientNet-b4
features at four depths, summed over channels, resized to SIZE, blurred, then scaled to [0, 1] over the whole batch.
The upstream repository must be on sys.path and be the working directory (it holds the config and the weights).
"""
import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.ndimage import gaussian_filter
from torchvision import transforms

SIZE = 224


def anomaly_maps(photos, renders):
    """photos: uint8 (N, 3, H, W). renders: (N, 3, H, W), floats in [0, 1] or uint8. Returns (N, SIZE, SIZE)."""
    from utils_pose_est import ModelHelper, update_config  # upstream code
    with open("PAD_utils/config_effnet.yaml") as f:
        cfg = update_config(EasyDict(yaml.load(f, Loader=yaml.FullLoader)))
    model = ModelHelper(cfg.net).eval().cuda()
    mse = torch.nn.MSELoss(reduction="none")
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def to_u8(r):
        if r.dtype == np.uint8:
            return r
        return (r.astype(np.float32).clip(0, 1) * 255).round().astype(np.uint8)

    maps = []
    with torch.no_grad():
        for p, r in zip(photos, renders):
            ref = tf(torch.from_numpy(to_u8(r))).unsqueeze(0).cuda()
            rgb = tf(torch.from_numpy(p)).unsqueeze(0).cuda()
            f_ref, f_rgb = model(ref), model(rgb)
            score = mse(ref, rgb).sum(1, keepdim=True)
            for a, b in zip(f_ref, f_rgb):
                score += torch.nn.functional.interpolate(
                    mse(a, b).sum(1, keepdim=True), size=SIZE, mode="bilinear", align_corners=False)
            maps.append(gaussian_filter(score.squeeze().cpu().numpy(), sigma=4))
    maps = np.stack(maps)
    return (maps - maps.min()) / (maps.max() - maps.min())
