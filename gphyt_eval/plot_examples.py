"""
Show predicted fields next to the truth for one test trajectory.

Rows: truth, pretrained GPhyT, fine-tuned GPhyT, |fine-tuned - truth|
Columns: last input frame, then rollout steps 1, 5, 10, 15.
Plotted quantity: vorticity (computed from the predicted velocity).

Usage:
    python plot_examples.py --dataset decay --family medium --stride 3 --traj 0
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C

STEPS = [1, 5, 10, 15]


def vorticity(u: np.ndarray, L: float) -> np.ndarray:
    """(..., n, n, 2) velocity -> (..., n, n) vorticity  ω = ∂x u_y - ∂y u_x."""
    n = u.shape[-2]
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    uh = np.fft.fft2(u, axes=(-3, -2))
    return np.real(np.fft.ifft2(1j * kx * uh[..., 1] - 1j * ky * uh[..., 0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="decay", choices=list(C.DOMAIN_EXTENT))
    p.add_argument("--size", default="S")
    p.add_argument("--family", default="medium", choices=C.FAMILIES)
    p.add_argument("--stride", type=int, default=3, choices=C.STRIDES)
    p.add_argument("--traj", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir, _, fig_dir = C.out_dirs(args.dataset, args.size)
    L = C.DOMAIN_EXTENT[args.dataset]

    vel = C.load_split(args.dataset, "test", args.family, args.traj + 1)[args.traj:args.traj + 1]
    ctx, tgt = C.make_clips(vel, args.stride)
    n_steps = max(STEPS)

    preds = {}
    for name, ckpt in [("pretrained", None), ("fine-tuned", run_dir / "best.pt")]:
        model = C.load_model(args.size, ckpt, device)
        preds[name] = C.rollout(model, torch.from_numpy(ctx).to(device), n_steps).cpu().numpy()[0]

    w_last = vorticity(ctx[0, -1], L)
    w_true = vorticity(tgt[0, :n_steps], L)
    w_pred = {n: vorticity(p_, L) for n, p_ in preds.items()}
    vmax = np.abs(w_true).max()

    rows = [("truth", w_true), ("pretrained", w_pred["pretrained"]), ("fine-tuned", w_pred["fine-tuned"]),
            ("|fine-tuned - truth|", np.abs(w_pred["fine-tuned"] - w_true))]
    fig, axes = plt.subplots(len(rows), len(STEPS) + 1, figsize=(3 * (len(STEPS) + 1), 3 * len(rows)))
    for r, (label, w) in enumerate(rows):
        err = label.startswith("|")
        ax = axes[r, 0]
        if r == 0:
            ax.imshow(w_last.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title("last input")
        ax.axis("off")
        for c, s in enumerate(STEPS, start=1):
            ax = axes[r, c]
            if err:
                im_err = ax.imshow(w[s - 1].T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
            else:
                im = ax.imshow(w[s - 1].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            l2 = np.linalg.norm(w[s - 1] - w_true[s - 1]) / np.linalg.norm(w_true[s - 1])
            ax.set_title(f"step {s}" if r == 0 else (f"step {s}" if err else f"step {s}  (err {l2:.2f})"))
            ax.axis("off")
        axes[r, 1].text(-0.15, 0.5, label, transform=axes[r, 1].transAxes, rotation=90,
                        va="center", ha="right", fontsize=12)
    fig.colorbar(im, ax=axes[:3, :].ravel().tolist(), shrink=0.6, label="vorticity")
    fig.colorbar(im_err, ax=axes[3, :].ravel().tolist(), shrink=0.8, label="abs. error")
    fig.suptitle(f"{args.dataset} test, {args.family} IC, stride {args.stride}, trajectory {args.traj}", fontsize=14)

    out = fig_dir / f"examples_{args.family}_stride{args.stride}_traj{args.traj}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
