"""
Fine-tune pretrained GPhyT on one Exponax dataset (train split, train-seen cells only).

Training samples: 4 input frames -> next frame, taken anywhere in a trajectory with the
frame stride of its IC family (simple:2, medium:3, complex:4, as the paper's diagonal).
Validation: rel. L2 averaged over a 10-step rollout on the val split, same cells. Best model is kept.

Usage:
    python finetune.py --dataset decay
    python finetune.py --dataset decay --steps 50        # quick test
"""

import argparse
import csv
import math
import time

import numpy as np
import torch

import common as C


def sample_batch(train: dict, batch_size: int, rng: np.random.Generator, device: str):
    xs, ys = [], []
    for _ in range(batch_size):
        fam = C.TRAIN_FAMILIES[rng.integers(len(C.TRAIN_FAMILIES))]
        vel = train[fam]
        s = C.TRAIN_STRIDE[fam]
        i = rng.integers(vel.shape[0])
        t0 = rng.integers(vel.shape[1] - C.N_IN * s)
        frames = vel[i, t0: t0 + (C.N_IN + 1) * s: s]            # (5, 64, 64, 2)
        xs.append(frames[: C.N_IN])
        ys.append(frames[C.N_IN:])
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.from_numpy(np.stack(ys)).to(device)
    return x, y


@torch.no_grad()
def validate(model, val: dict, device: str) -> dict:
    model.eval()
    out = {}
    for fam in C.TRAIN_FAMILIES:
        ctx, tgt = C.make_clips(val[fam], C.TRAIN_STRIDE[fam])
        pred = C.rollout(model, torch.from_numpy(ctx).to(device), C.N_HORIZON).cpu().numpy()
        out[fam] = float(C.rel_l2(pred, tgt[:, : C.N_HORIZON]).mean())
    out["mean"] = float(np.mean([out[f] for f in C.TRAIN_FAMILIES]))
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="decay", choices=list(C.DOMAIN_EXTENT))
    p.add_argument("--size", default="S", choices=["S", "M", "L", "XL"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--n_val_traj", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run_dir = C.RUNS_DIR / f"{args.dataset}_gphyt-{args.size}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("loading data ...")
    train = {f: C.load_split(args.dataset, "train", f) for f in C.TRAIN_FAMILIES}
    val = {f: C.load_split(args.dataset, "val", f, args.n_val_traj) for f in C.TRAIN_FAMILIES}
    print({f: v.shape for f, v in train.items()})

    model = C.load_model(args.size, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup)
        * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))

    v = validate(model, val, device)
    print(f"step 0 (pretrained, zero-shot)  val rel L2 (mean over steps 1-10): {v}")
    best = v["mean"]
    torch.save(model.state_dict(), run_dir / "best.pt")

    log = open(run_dir / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train_loss", "val_mean", *C.TRAIN_FAMILIES, "seconds"])
    writer.writerow([0, "", v["mean"], *[v[f] for f in C.TRAIN_FAMILIES], 0])

    t_start, running = time.time(), []
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train, args.batch_size, rng, device)
        xg, mean, std = C.normalize(C.to_gphyt(x))
        yg = (C.to_gphyt(y) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(xg)
        loss = torch.nn.functional.mse_loss(pred.float()[..., 3:], yg[..., 3:])   # velocity fields

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        running.append(loss.item())

        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, val, device)
            tl = float(np.mean(running)); running = []
            el = time.time() - t_start
            flag = ""
            if v["mean"] < best:
                best = v["mean"]
                torch.save(model.state_dict(), run_dir / "best.pt")
                flag = "  <- best, saved"
            print(f"step {step:>5}  train loss {tl:.4f}  val rel L2 {v['mean']:.4f} "
                  f"({', '.join(f'{f} {v[f]:.3f}' for f in C.TRAIN_FAMILIES)})  {el/60:.1f} min{flag}")
            writer.writerow([step, tl, v["mean"], *[v[f] for f in C.TRAIN_FAMILIES], round(el)])
            log.flush()

    torch.save(model.state_dict(), run_dir / "last.pt")
    log.close()
    print(f"done. best val rel L2 = {best:.4f}. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
