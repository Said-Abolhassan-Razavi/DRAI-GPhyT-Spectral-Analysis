# DRAI – GPhyT Spectral Analysis

Spectral evaluation of the **General Physics Transformer (GPhyT)**
([paper](https://arxiv.org/abs/2509.13805) · [code](https://github.com/FloWsnr/General-Physics-Transformer) ·
[weights](https://huggingface.co/flwi/Physics-Foundation-Model)) on PDE datasets generated with
[Exponax](https://github.com/Ceyron/exponax), following Appendix A of
*"Do Physics Foundation Models Learn Generalizable Physics?"* (arXiv:2605.29283).

Part of the DRAI course project *Spectral Analysis on Foundation Models for PDEs*.

**Question:** does GPhyT reproduce the energy spectrum of the reference solutions, especially at high
wavenumbers, and how does this depend on initial-condition complexity, time stride and rollout horizon?

## Repository structure

```
gphyt_eval/
├── common.py          data loading, GPhyT input format, model loading, rollout, metrics
├── finetune.py        fine-tune pretrained GPhyT on the train split
├── evaluate.py        5x5 test grid: rel. L2 error + energy spectra, figures
├── plot_examples.py   predicted vs true fields for one test trajectory
└── runs/decay_gphyt-S/
    ├── log.csv        training log
    └── eval/          summary.csv + figures
```

Data files (`*.npz`) and model checkpoints (`*.pt`) are not in the repository.

## Setup

```bash
pip install torch numpy matplotlib einops huggingface_hub
git clone https://github.com/FloWsnr/General-Physics-Transformer
```

Then edit the paths at the top of `gphyt_eval/common.py`:
- `GPHYT_ROOT`: the cloned General-Physics-Transformer folder
- `DATA_DIR`: the folder with the dataset files (`decay.npz`, `burgers.npz`, ...)

The pretrained weights are downloaded automatically from Hugging Face. A GPU is recommended
(the Decay run below took ~12 min of fine-tuning on an RTX 5060 Laptop GPU).

## Usage

```bash
cd gphyt_eval
python finetune.py --dataset decay                                  # fine-tune GPhyT-S
python evaluate.py --dataset decay                                  # full test grid
python plot_examples.py --dataset decay --family medium --stride 3  # field images
```

Datasets supported: `decay`, `kolmogorov` (stored as vorticity, converted to velocity) and `burgers` (velocity).

## Method

- **Data:** 2D periodic, 64×64, 100 frames per trajectory; 5 IC families
  (OOD-simple, simple, medium, complex, OOD-complex); 200 train / 100 val / 100 test trajectories per family.
- **GPhyT input:** 4 frames of velocity. The periodic 64×64 field is tiled 4×2 to GPhyT's fixed 256×128 grid
  (no interpolation, so the spectrum is unchanged). Pressure, density and temperature channels are set to zero.
  Per-sample z-score normalisation.
- **Fine-tuning:** GPhyT-S, train-seen cells only (simple/stride 2, medium/stride 3, complex/stride 4),
  1-step MSE on velocity, AdamW (lr 5e-5), 5000 steps, best checkpoint by validation error.
- **Evaluation (paper protocol):** 20-frame clips with frame stride 1–5; 4 context frames, then
  10 in-horizon + 5 OOD rollout steps; all 5 IC families × 5 strides. Baseline: copy the last frame.
- **Spectrum:** radial kinetic-energy spectrum E(k); high-k band = 11 ≤ k ≤ 21
  (the solver's 2/3 dealiasing keeps |k| ≤ 21).

## First results (Decay, GPhyT-S)

Relative L2 error of velocity at rollout step 10, averaged by test-cell type:

| Test cells | Copy last frame | Pretrained | Fine-tuned |
|---|---|---|---|
| train-seen | 0.76 | 0.76 | **0.52** |
| in-range shift | 0.77 | 0.78 | **0.55** |
| IC-OOD | 0.67 | 0.75 | **0.55** |
| dynamic-OOD | 0.66 | 0.71 | **0.59** |
| joint-OOD | 0.58 | 0.70 | 0.58 |

Observations:
- Zero-shot, GPhyT is no better than copying the last frame; fine-tuning lowers the error clearly.
- Large scales (k ≤ 4) are reproduced well, but both models put **too much** energy at high wavenumbers
  (k > 10), growing with each rollout step. Fine-tuning lowers the L2 error but makes the high-k excess worse.
- In vorticity the fine-tuned prediction is accurate at step 1 (error 0.07) but shows **16×16 block
  artifacts** from GPhyT's patch tokenizer after ~5 steps, which dominate by step 10.

Limitations: one model size, one dataset, one training run; the high-k ratio is unstable for OOD-simple
(almost no true high-k energy); the effect of tiling and of the zero-filled channels has not been tested.

## Next steps

- Run the same pipeline on Burgers and Kolmogorov.
- Relate the high-k error to the L2 error, IC complexity and rollout horizon.
