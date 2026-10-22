# Basic 1D CNN vs 1D ResNet for Dense QAM

EE136IU Capstone Mini-Project, Topic 5. Can residual connections (y = F(x) + x) reduce
16QAM / 64QAM confusion compared with a plain 1D CNN of matched parameter count?

## Quick start

```bash
pip install -r requirements.txt
python run_all.py --quick        # small dataset + pretrained checkpoints + eval, < 30 s
streamlit run demo/app.py        # QAM Test Bench
```

Full pipeline:

```bash
python -m dataset.generator      # writes data/{train,val,test}.npz (~5 s)
python -m pytest -q              # shape, RMS = 1, no leakage between splits
python -m dataset.sanity_plots   # constellations, measured SNR, C42 -> results/figures
python train.py --model resnet1d --seed 0   # one run
python train.py --all                       # 3 models x 3 seeds (see kaggle/HUONG_DAN_KAGGLE.md)
python evaluate.py               # acc/F1 vs SNR, confusion matrices, QAM sub-block, McNemar -> results/
```

## Layout

| Path | Contents |
| --- | --- |
| `config.yaml` | All locked settings: classes, SNR grid, sps, split seeds, model widths, training |
| `dataset/` | Burst generator (symbols, RRC, channel, RMS norm) and PyTorch dataset |
| `models/` | `cnn_base.py`, `cnn_plain_deep.py`, `resnet1d.py` |
| `utils/` | Seeds, config loader, metrics, plotting |
| `demo/` | Streamlit app |
| `results/` | Figures and CSV tables |
| `checkpoints/` | Best checkpoint per model and seed |
| `tests/` | Shape, RMS, no-leakage and matched-parameter checks |
| `kaggle/` | How to run the training on a Kaggle GPU |

## Setup locked in config.yaml

- 8 classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, CPFSK, GFSK
- Bursts of 128 samples, sps = 4 (32 symbols), RRC rolloff 0.35
- SNR -10 to +18 dB in 2 dB steps (15 points)
- Random phase, CFO up to 0.4% of fs, AWGN, per-burst RMS normalization
- 1000 bursts per (class, SNR), split 70/15/15 with a separate seed per split
- Three models at about 150k parameters each (CNN-Base 154.7k, CNN-Deep-Plain 144.6k, ResNet1D 152.6k), 3 training seeds per model
