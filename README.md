# Code for the paper "Universal interpolation for deep residual self-attention networks", Sibylle Marcotte & Joan Bruna 2026.

## Head-width threshold experiment

Code accompanying Appendix M. See the appendix for the protocol and results.
Requires Python 3.10+; experiments used Python 3.12.14 and PyTorch 2.14.0.

### Install

Run inside this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Reproduce

Main experiment and figures:

```sh
python run_threshold.py
python plot_threshold.py results_threshold/results.json
```

Experiment at D=96:

```sh
python run_threshold.py \
  --pairs 3:1 --shapes 16:2 8:4 4:8 --seeds 0 \
  --initializations fan_in --offsets 0 --max-steps 5 \
  --output results_threshold_large
```

Each run saves `results.json` and `summary.csv` in its output folder;
the plotting script adds PNG/PDF figures.
Use `python run_threshold.py --help` for other settings.
