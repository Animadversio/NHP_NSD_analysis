"""DINOv2 and ResNet50 feature extraction for NSD stimuli."""

import os
import numpy as np
import torch

FEAT_CACHE_DEFAULT = os.path.join(
    os.path.dirname(__file__), '..', 'notebooks', 'cache', 'dinov2_nsd_features.pkl'
)
N_BLOCKS = 12


def load_dinov2_features(cache_path: str = FEAT_CACHE_DEFAULT) -> dict:
    """
    Load cached DINOv2 ViT-B/14-reg features (1072 images × 12 blocks × 2 types).

    Returns dict: {'blocks.{i}_cls': (1072, 768), 'blocks.{i}_patch': (1072, 768), ...}
    """
    import pickle
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def extract_dinov2_features(
    images: torch.Tensor,
    device: str = 'cuda',
    n_blocks: int = N_BLOCKS,
    batch_size: int = 64,
) -> dict:
    """
    Extract DINOv2 ViT-B/14-reg features from a batch of images.

    Parameters
    ----------
    images : (N, 3, H, W) float tensor, values in [0, 1]
    device : 'cuda' or 'cpu'
    n_blocks : number of transformer blocks to hook (12 for ViT-B)

    Returns
    -------
    dict: {'blocks.{i}_cls': (N, 768), 'blocks.{i}_patch': (N, 768)}
    """
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = model.to(device).eval()

    n_registers = 4  # ViT-B/14-reg has 4 register tokens
    hooks: dict[str, list] = {f'blocks.{i}': [] for i in range(n_blocks)}
    handles = []
    for i in range(n_blocks):
        def make_hook(name):
            def hook(mod, inp, out):
                hooks[name].append(out.detach().cpu())
            return hook
        handles.append(model.blocks[i].register_forward_hook(make_hook(f'blocks.{i}')))

    all_feats: dict[str, list] = {f'blocks.{i}_{t}': [] for i in range(n_blocks)
                                   for t in ('cls', 'patch')}
    images = images.to(device)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            for k in hooks:
                hooks[k].clear()
            model(batch)
            for i in range(n_blocks):
                feat = hooks[f'blocks.{i}'][0]          # (B, 1+n_reg+n_patches, D)
                all_feats[f'blocks.{i}_cls'].append(feat[:, 0, :].numpy())
                all_feats[f'blocks.{i}_patch'].append(
                    feat[:, 1 + n_registers:, :].mean(dim=1).numpy()
                )

    for h in handles:
        h.remove()

    return {k: np.concatenate(v, axis=0) for k, v in all_feats.items()}


def extract_resnet50_features(
    images: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 64,
) -> dict:
    """
    Extract ResNet50 intermediate features.

    Returns
    -------
    dict: {'relu': (N, D), 'layer1': ..., 'layer2', 'layer3', 'layer4', 'avgpool'}
    """
    import torchvision.models as models
    model = models.resnet50(pretrained=True).to(device).eval()

    layer_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    hooks: dict[str, list] = {n: [] for n in layer_names}
    handles = []
    for name in layer_names:
        layer = getattr(model, name)
        def make_hook(n):
            def hook(mod, inp, out):
                feat = out.detach().cpu()
                if feat.dim() > 2:
                    feat = feat.mean(dim=[2, 3])     # global average pool
                hooks[n].append(feat.numpy())
            return hook
        handles.append(layer.register_forward_hook(make_hook(name)))

    all_feats = {n: [] for n in layer_names}
    images = images.to(device)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            for n in layer_names:
                hooks[n].clear()
            model(images[start:start + batch_size])
            for n in layer_names:
                all_feats[n].append(hooks[n][0])

    for h in handles:
        h.remove()

    return {k: np.concatenate(v, axis=0) for k, v in all_feats.items()}
