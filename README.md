# ML Learning

A multi-framework machine learning practice repository using UV workspaces.

## Structure

```
ml-learning/
├── packages/
│   ├── pytorch-learning/    # PyTorch practice workspace
│   │   ├── pyproject.toml
│   │   └── src/pytorch_learning/
│   └── pyro-learning/       # Pyro probabilistic programming workspace
│       ├── pyproject.toml
│       └── src/pyro_learning/
├── scripts/
│   ├── pytorch/             # PyTorch practice scripts
│   └── pyro/                # Pyro practice scripts
├── pyproject.toml           # Root workspace configuration
└── flake.nix                # Nix dev shells
```

## Development Shells

This project uses Nix flakes for reproducible development environments.

### Default Shell (CPU-only)

```bash
nix develop
```

Lightweight shell with basic tooling (uv, git, node).

### CUDA Shell (GPU support)

```bash
nix develop .#cuda
```

Full CUDA-enabled shell with GPU support, OpenCV, and additional tooling.

## Installation

```bash
# Enter the dev shell
nix develop

# Install all dependencies
uv sync
```

## Workspaces

### pytorch-learning

PyTorch deep learning practice. Scripts in `scripts/pytorch/`.

CLI available after installation:

```bash
pytorch-learn --help
pytorch-learn info
```

### pyro-learning

Pyro probabilistic programming practice. Scripts in `scripts/pyro/`.

CLI available after installation:

```bash
pyro-learn --help
pyro-learn info
```
