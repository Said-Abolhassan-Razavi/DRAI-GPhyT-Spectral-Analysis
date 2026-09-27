"""
Evaluate GPhyT (pretrained zero-shot and fine-tuned) on the full 5x5 test grid
(IC family x frame stride) of one dataset, and compute energy spectra.

For each cell: 4 context frames -> 15-step rollout (10 in-horizon + 5 OOD rollout).
Saves:
    runs/<dataset>_gphyt-<size>/eval/results.npz   rel. L2 and spectra for every cell/step
    runs/<dataset>_gphyt-<size>/eval/summary.csv   one row per model x cell
    runs/<dataset>_gphyt-<size>/eval/*.png         figures

Usage:
    python evaluate.py --dataset decay
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C

# Wavenumber bands for the spectral summary. The solver's 2/3 dealiasing keeps |k| <= 21 on a 64 grid.
BANDS = {"low": (1, 4), "mid": (5, 10), "high": (11, 21)}
N_STEPS = C.N_HORIZON + C.N_OOD


def cell_type(fam: str, stride: int) -> str:
    if C.TRAIN_STRIDE.get(fam) == stride:
        return "train-seen"
    ic_ood = fam.startswith("OOD")
    dyn_ood = stride in (1, 5)
    if ic_ood and dyn_ood:
        return "joint-OOD"
    if ic_ood:
        return "IC-OOD"
    if dyn_ood:
        return "dynamic-OOD"
    return "in-range shift"


def band_ratio(E_pred, E_true, band):
    lo, hi = BANDS[band]
    return E_pred[..., lo:hi + 1].sum(-1) / E_true[..., lo:hi + 1].sum(-1)


def run_model(model, test, device, batch=50):
    """Returns dict[(fam, stride)] -> (rel_l2 (N,S), E_pred (S,nk) mean over N, E_true (S,nk))."""
    out = {}
    for fam in C.FAMILIES:
        for s in C.STRIDES:
            ctx, tgt = C.make_clips(test[fam], s)
            pred = np.concatenate([
                C.rollout(model, torch.from_numpy(ctx[i:i + batch]).to(device), N_STEPS).cpu().numpy()
                for i in range(0, len(ctx), batch)])
            out[(fam, s)] = (C.rel_l2(pred, tgt),
                             C.energy_spectrum(pred).mean(0),
                             C.energy_spectrum(tgt).mean(0))
        print(f"  {fam} done")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="decay", choices=list(C.DOMAIN_EXTENT))
    p.add_argument("--size", default="S", choices=["S", "M", "L", "XL"])
    p.add_argument("--checkpoint", default="best.pt", help="file inside the run dir")
    p.add_argument("--max_traj", type=int, default=None, help="limit test trajectories per family")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.RUNS_DIR / f"{args.dataset}_gphyt-{args.size}"
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading test data ...")
    test = {f: C.load_split(args.dataset, "test", f, args.max_traj) for f in C.FAMILIES}

    results = {}
    for name, ckpt in [("pretrained", None), ("finetuned", run_dir / args.checkpoint)]:
        print(f"evaluating {name} ...")
        results[name] = run_model(C.load_model(args.size, ckpt, device), test, device)

    # copy-last-frame baseline
    baseline = {}
    for fam in C.FAMILIES:
        for s in C.STRIDES:
            ctx, tgt = C.make_clips(test[fam], s)
            baseline[(fam, s)] = C.rel_l2(np.repeat(ctx[:, -1:], N_STEPS, axis=1), tgt)

    # ---------------- save raw results ----------------
    save = {}
    for name, res in list(results.items()) + [("copy_last", None)]:
        for (fam, s) in [(f, s) for f in C.FAMILIES for s in C.STRIDES]:
            key = f"{name}/{fam}/stride{s}"
            if res is None:
                save[f"{key}/rel_l2"] = baseline[(fam, s)]
            else:
                l2, Ep, Et = res[(fam, s)]
                save[f"{key}/rel_l2"] = l2
                save[f"{key}/E_pred"] = Ep
                save[f"{key}/E_true"] = Et
    np.savez_compressed(out_dir / "results.npz", **save)

    # ---------------- summary table ----------------
    rows = []
    for name in ["copy_last", *results]:
        for fam in C.FAMILIES:
            for s in C.STRIDES:
                l2 = baseline[(fam, s)] if name == "copy_last" else results[name][(fam, s)][0]
                row = dict(model=name, family=fam, stride=s, cell=cell_type(fam, s),
                           l2_step1=l2[:, 0].mean(), l2_step5=l2[:, 4].mean(),
                           l2_step10=l2[:, 9].mean(), l2_step15=l2[:, 14].mean())
                if name != "copy_last":
                    _, Ep, Et = results[name][(fam, s)]
                    for b in BANDS:
                        row[f"{b}_ratio_step10"] = band_ratio(Ep[9], Et[9], b)
                rows.append(row)
    keys = list(rows[-1].keys())
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------------- figures ----------------
    def grid(metric):
        return {n: np.array([[metric(n, f, s) for s in C.STRIDES] for f in C.FAMILIES]) for n in results}

    def heatmaps(data, title, fname, cmap, vmin=None, vmax=None, center=None):
        fig, axes = plt.subplots(1, len(data), figsize=(5.5 * len(data), 4.4))
        for ax, (n, g) in zip(axes, data.items()):
            if center is not None:
                d = np.abs(np.log10(g)).max()
                im = ax.imshow(np.log10(g), cmap=cmap, vmin=-d, vmax=d)
                label = "log10 ratio (pred / true)"
            else:
                im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
                label = title
            for i in range(len(C.FAMILIES)):
                for j in range(len(C.STRIDES)):
                    txt = f"{g[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            fontweight="bold" if cell_type(C.FAMILIES[i], C.STRIDES[j]) == "train-seen" else None)
            ax.set_xticks(range(len(C.STRIDES)), [f"stride {s}" for s in C.STRIDES])
            ax.set_yticks(range(len(C.FAMILIES)), C.FAMILIES)
            ax.set_title(n)
            fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        fig.suptitle(f"{args.dataset}, GPhyT-{args.size}: {title} (bold = train-seen)")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    l2g = grid(lambda n, f, s: results[n][(f, s)][0][:, 9].mean())
    heatmaps(l2g, "rel. L2 at step 10", "grid_rel_l2_step10.png", "viridis",
             vmin=0, vmax=max(g.max() for g in l2g.values()))
    hfg = grid(lambda n, f, s: band_ratio(results[n][(f, s)][1][9], results[n][(f, s)][2][9], "high"))
    heatmaps(hfg, "high-k energy ratio (k 11-21) at step 10", "grid_high_k_ratio_step10.png",
             "RdBu_r", center=1.0)

    # spectra for the three train-seen cells
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, fam in zip(axes, C.TRAIN_FAMILIES):
        s = C.TRAIN_STRIDE[fam]
        Et = results["pretrained"][(fam, s)][2]
        k = np.arange(1, Et.shape[1])
        ax.loglog(k, Et[9, 1:], "k", lw=2, label="truth")
        for n, st in zip(results, ["--", "-"]):
            ax.loglog(k, results[n][(fam, s)][1][9, 1:], st, label=n)
        ax.axvline(21, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{fam}, stride {s}, step 10")
        ax.set_xlabel("k"); ax.set_ylabel("E(k)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spectra_train_cells_step10.png", dpi=120)
    plt.close(fig)

    # spectral ratio vs k across the rollout (medium, stride 3), fine-tuned
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, n in zip(axes, results):
        _, Ep, Et = results[n][("medium", 3)]
        k = np.arange(1, 22)
        for st in [0, 4, 9, 14]:
            ax.semilogy(k, Ep[st, 1:22] / Et[st, 1:22], label=f"step {st + 1}")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_title(f"{n}: E_pred / E_true (medium, stride 3)")
        ax.set_xlabel("k")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_ratio_vs_step.png", dpi=120)
    plt.close(fig)

    # ---------------- console summary ----------------
    print("\nmean rel. L2 at step 10, by cell type:")
    types = ["train-seen", "in-range shift", "IC-OOD", "dynamic-OOD", "joint-OOD"]
    print(f"{'model':<12}" + "".join(f"{t:>16}" for t in types))
    for name in ["copy_last", *results]:
        vals = [np.mean([r["l2_step10"] for r in rows if r["model"] == name and r["cell"] == t]) for t in types]
        print(f"{name:<12}" + "".join(f"{v:>16.3f}" for v in vals))
    print("\nhigh-k energy ratio (pred/true, k 11-21) at step 10, by cell type:")
    for name in results:
        vals = [np.mean([r["high_ratio_step10"] for r in rows if r["model"] == name and r["cell"] == t]) for t in types]
        print(f"{name:<12}" + "".join(f"{v:>16.3f}" for v in vals))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
