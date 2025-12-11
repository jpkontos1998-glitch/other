# Ataraxos

GPU-accelerated self-play reinforcement learning and test-time search for Stratego.

## Requirements

- Linux
- GPU supporting CUDA 11.8
- Software and associated versions detailed in `environment.yml`

## Installation

```bash
micromamba env create -f environment.yml
micromamba activate ataraxos
make
pip install -e .
```

A typical installation time is a few minutes.

## Demo

```bash
python scripts/train/rl_main.py --num_envs 128 --rl.dtype float32 --rl.torch_compile false
```

Logs training statistics to stdout. Iterations usually take 10–20 seconds on an RTX A6000.

## Usage

**Train a policy:**
```bash
python scripts/train/rl_main.py --help
```

**Train a belief model:**
```bash
python scripts/train/belief_main.py --help
```
