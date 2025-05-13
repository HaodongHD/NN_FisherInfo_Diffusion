# Diffusion of the Fisher Information in Structured Deep RNNs

This repository accompanies the NeurIPS 2025 submission:

**"Diffusion of the Fisher Information in a Large Structured Deep RNN"**
*Haodong Qin, Tatyana Sharpee*
\[UC San Diego, Salk Institute]

---

## Overview

This project investigates the encoding and retention of information in recurrent neural networks (RNNs) with structured connectivity. It introduces a novel theoretical and computational framework based on **Fisher information diffusion** to:

* Analyze how information propagates across subpopulations of neurons
* Derive optimal network configurations that retain information over long timescales
* Connect random matrix theory, dynamical systems, and compressed sensing

We also validate the theoretical insights using real-world CIFAR-10 stimuli.

---

## Highlights

* **Fisher Diffusion Operator**: Analytical tool that models temporal information propagation in structured RNNs.
* **Structured Optimization**: Explore Toeplitz and generalized Toeplitz architectures for maximizing Fisher information.
* **Differential Evolution (DE)**: Utilized for optimizing complex, non-convex network structures.
* **Torch-accelerated Simulation**: Fast simulations of large RNNs using PyTorch backend.
* **Natural Image Task**: Network tested for geometric fidelity using CIFAR-10 inputs.

---

## Installation

```bash
pip install numpy scipy matplotlib torch pandas
```

To use MPS on macOS (Apple Silicon), install PyTorch with MPS support:

```bash
pip install torch torchvision torchaudio
```

---

## Files

| File                  | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| `NN_analytic_opt.py`  | Optimization routines for structured connectivity using DE and gradient-free methods |
| `NN_analytic_test.py` | Analytical mean-field calculations of Fisher information and matrix eigenvalues      |
| `NN_sim_test.py`      | Torch-based simulation of recurrent neural dynamics, empirical Fisher estimation     |

---

## Quick Start

```python
# Run a full optimization loop
python run_optimize.py  # replace with your entrypoint script if separate
```

This includes:

* Constructing Toeplitz/generalized Toeplitz matrices
* Optimizing their parameters
* Simulating information propagation
* Visualizing heatmaps and Fisher dynamics

---

## Core Components

### 1. **Optimization (NN\_analytic\_opt.py)**

* Defines objective functions for different matrix classes
* Uses `scipy.optimize.minimize` and `differential_evolution`
* Tracks alignment and spectral conditions (e.g., spectral radius \~1)

### 2. **Mean-Field Theory (NN\_analytic\_test.py)**

* Computes Fisher diffusion matrix $A$
* Calculates population-wise Fisher information over time
* Solves self-consistent equations for subpopulation variances

### 3. **Simulation (NN\_sim\_test.py)**

* Empirically simulates RNN dynamics
* Computes trajectory-level Fisher information
* Evaluates geometry preservation via Shepard diagrams
* Leverages `torch` for parallel multi-trajectory rollout

---

## Real-Image Evaluation

The `simulation_multiple_stimulus_multiPopu_torch` method evaluates how well the network preserves the geometry of natural image inputs (CIFAR-10) across time. Outputs are Pearson correlation curves of pairwise distances between original and latent representations.

---

## Citation

```bibtex
@article{qin2025diffusion,
  title={Diffusion of the Fisher Information in a Large Structured Deep RNN},
  author={Qin, Haodong and Sharpee, Tatyana},
  journal={NeurIPS},
  year={2025}
}
```

---

## Contact

Haodong Qin — [qhaodong@ucsd.edu](mailto:qhaodong@ucsd.edu)

---

## License

MIT License
