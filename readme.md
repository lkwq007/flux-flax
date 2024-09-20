# FLUX-Flax

A JAX port of FLUX.1 models using `flax.nnx`. 

![img](https://github.com/user-attachments/assets/77a9bf56-8da7-434d-9faf-1510a33e63dd)

## Status

Only tested with GPU now. 

Currently no quantization support & no torch-like CPU offloading support. 

PRs are welcome.

## Local installation

```bash
git clone https://github.com/lkwq007/flux-flax.git
cd flux-flax
mamba create -p ./env python=3.10
mamba activate ./env
pip install -r requirements.txt
```

## Usage

For interactive sampling run

```bash
python main.py --name <name>
```

Or to generate a single sample run (not recommended, as jit compilation takes time)

```bash
python main.py --name <name> \
  --height <height> --width <width> --nonloop \
  --prompt "<prompt>"
```
