"""
Shared helpers for running GPhyT on the Exponax datasets (decay / burgers / kolmogorov).

- loading a split from the .npz and converting it to velocity (u_x, u_y)
- packing velocity into GPhyT's input format (tiling 64x64 -> 256x128, 5 fields)
- building GPhyT and loading pretrained / fine-tuned weights
- autoregressive rollout, relative L2 error and radial energy spectrum
"""

import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths (edit here if the folders move)
# ---------------------------------------------------------------------------
DRAI_DIR = Path(r"D:\Universtiy of Paris Saclay\M2 courses\DRAI")
GPHYT_ROOT = DRAI_DIR / "General-Physics-Transformer-main" / "General-Physics-Transformer-main"
DATA_DIR = DRAI_DIR / "General-Physics-Transformer-main"   # where the .npz files are
RUNS_DIR = Path(__file__).resolve().parent / "runs"

sys.path.insert(0, str(GPHYT_ROOT))
from gphyt.models.transformer.model import get_model  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset settings (from gen_all.py / Appendix A)
# ---------------------------------------------------------------------------
DOMAIN_EXTENT = {"decay": 5.0, "kolmogorov": 6.0, "burgers": 3.0}
STORES_VORTICITY = {"decay": True, "kolmogorov": True, "burgers": False}

FAMILIES = ["OOD-simple", "simple", "medium", "complex", "OOD-complex"]
TRAIN_FAMILIES = ["simple", "medium", "complex"]
STRIDES = [1, 2, 3, 4, 5]                       # Dynamic-OOD-small ... Dynamic-OOD-large
# train-seen (diagonal) cells of the paper: IC family -> frame stride
TRAIN_STRIDE = {"simple": 2, "medium": 3, "complex": 4}

# Evaluation clip (paper): 20 frames = 5 input + 10 in-horizon + 5 OOD rollout.
# GPhyT takes 4 input frames, so it uses the last 4 of the 5 input frames.
CLIP_LEN, N_CONTEXT, N_HORIZON, N_OOD = 20, 5, 10, 5
N_IN = 4

# GPhyT input geometry
IMG_T, IMG_H, IMG_W, N_FIELDS = 4, 256, 128, 5
GRID = 64
TILE = (IMG_H // GRID, IMG_W // GRID)            # (4, 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def vorticity_to_velocity(w: np.ndarray, L: float) -> np.ndarray:
    """(..., n, n) vorticity -> (..., n, n, 2) velocity, periodic, via Δψ = ω, u = (-∂yψ, ∂xψ)."""
    n = w.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")    # axis -2 = x, axis -1 = y (Exponax convention)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    psi_hat = -np.fft.fft2(w) / k2
    psi_hat[..., 0, 0] = 0
    ux = np.real(np.fft.ifft2(-1j * ky * psi_hat))
    uy = np.real(np.fft.ifft2(1j * kx * psi_hat))
    return np.stack([ux, uy], axis=-1).astype(np.float32)


def load_split(dataset: str, split: str, family: str, max_traj: int | None = None) -> np.ndarray:
    """Return velocity trajectories of shape (N, T, 64, 64, 2)."""
    arr = np.load(DATA_DIR / f"{dataset}.npz")[f"{split}_{family}"]   # (N, T, C, H, W)
    if max_traj is not None:
        arr = arr[:max_traj]
    if STORES_VORTICITY[dataset]:
        return vorticity_to_velocity(arr[:, :, 0], DOMAIN_EXTENT[dataset])
    return np.moveaxis(arr[:, :, :2], 2, -1).astype(np.float32)       # burgers: (u_x, u_y)


def make_clips(vel: np.ndarray, stride: int):
    """Paper protocol: clip = frames 0, s, 2s, ..., 19s.
    Returns context (N, 4, ...) = clip[1:5] and target (N, 15, ...) = clip[5:20]."""
    clip = vel[:, : stride * CLIP_LEN : stride]
    return clip[:, N_CONTEXT - N_IN:N_CONTEXT], clip[:, N_CONTEXT:]


def to_gphyt(vel: torch.Tensor) -> torch.Tensor:
    """(B, T, 64, 64, 2) -> (B, T, 256, 128, 5).

    The periodic 64x64 field is tiled 4x2 (no interpolation, spectrum unchanged).
    Fields follow GPhyT's order (pressure, density, temperature, vel_x, vel_y);
    the three missing fields are left at zero.
    """
    v = vel.repeat(1, 1, TILE[0], TILE[1], 1)
    pad = torch.zeros(v.shape[:-1] + (N_FIELDS - 2,), dtype=v.dtype, device=v.device)
    return torch.cat([pad, v], dim=-1)


def from_gphyt(x: torch.Tensor) -> torch.Tensor:
    """(B, T, 256, 128, 5) -> (B, T, 64, 64, 2): first tile, velocity fields only."""
    return x[:, :, :GRID, :GRID, 3:]


def normalize(x: torch.Tensor):
    """Per-sample, per-field z-score over the input window (as GPhyT's instance norm)."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    return (x - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(size: str = "S") -> torch.nn.Module:
    cfg = {
        "img_size": (IMG_T, IMG_H, IMG_W),
        "transformer": dict(input_channels=N_FIELDS, model_size=f"GPT_{size}", att_mode="full",
                            dropout=0.0, pos_enc_mode="absolute", patch_size=(1, 16, 16),
                            stochastic_depth_rate=0.0, use_derivatives=True, integrator="Euler"),
        "tokenizer": dict(tokenizer_mode="linear", detokenizer_mode="linear",
                          tokenizer_overlap=0, detokenizer_overlap=0),
    }
    return get_model(cfg)


def load_pretrained(model: torch.nn.Module, size: str = "S") -> None:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("flwi/Physics-Foundation-Model", f"gphyt-{size}.pth")
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict({k.replace("module._orig_mod.", ""): v for k, v in sd.items()})


def load_model(size: str = "S", checkpoint: str | Path | None = None, device: str = "cuda"):
    """Pretrained weights if `checkpoint` is None, otherwise a fine-tuned state_dict."""
    model = build_model(size)
    if checkpoint is None:
        load_pretrained(model, size)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def rollout(model, context: torch.Tensor, n_steps: int, amp: bool = True) -> torch.Tensor:
    """context: (B, 4, 64, 64, 2) velocity -> prediction (B, n_steps, 64, 64, 2)."""
    x, mean, std = normalize(to_gphyt(context))
    preds = []
    for _ in range(n_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = model(x).float()
        preds.append(y)
        x = torch.cat([x[:, 1:], y], dim=1)
    return from_gphyt(torch.cat(preds, dim=1) * std + mean)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(N, S, H, W, C) -> (N, S) relative L2 error per trajectory and step."""
    d = (pred - true).reshape(*pred.shape[:2], -1)
    t = true.reshape(*true.shape[:2], -1)
    return np.linalg.norm(d, axis=-1) / np.linalg.norm(t, axis=-1)


_K_BINS = None


def energy_spectrum(u: np.ndarray) -> np.ndarray:
    """Radial kinetic-energy spectrum. u: (..., n, n, 2) -> (..., n_k), k = 0 .. n*sqrt(2)/2."""
    global _K_BINS
    n = u.shape[-2]
    if _K_BINS is None or _K_BINS.shape[0] != n:
        f = np.fft.fftfreq(n) * n
        _K_BINS = np.rint(np.sqrt(f[:, None] ** 2 + f[None] ** 2)).astype(int)
    uh = np.fft.fft2(u, axes=(-3, -2)) / n**2
    e = 0.5 * (np.abs(uh) ** 2).sum(-1)                           # (..., n, n)
    flat = e.reshape(-1, n * n)
    nk = _K_BINS.max() + 1
    out = np.stack([np.bincount(_K_BINS.ravel(), weights=row, minlength=nk) for row in flat])
    return out.reshape(*e.shape[:-2], nk)
