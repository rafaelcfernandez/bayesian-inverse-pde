# Bayesian Solutions to Inverse Problems in Differential Equations

This repository contains the implementation of a unified Bayesian framework for solving inverse problems in differential equations, as described in our paper "Bayesian Solutions to Inverse Problems in Differential Equations: From Theory to Implementation."

## Overview

We present two complementary approaches for parameter estimation in differential equations:

1. **Non-linear regression method**: Using exact analytical solutions or numerical approximations
2. **Gaussian process surrogate models**: A meshless approach that avoids discretization errors

Both methods are implemented within a fully Bayesian framework using:
- Hamiltonian Monte Carlo (HMC) with No-U-Turn Sampler
- Automatic Differentiation Variational Inference (ADVI)

## Repository Structure

bayesian-inverse-pde/
├── README.md
├── LICENSE
├── inference/
│   ├── stan_models/
│   │   ├── nonlinear_regression.stan  # Stan model for regression approach
│   │   └── gaussian_process.stan      # Stan model for GP approach
│   ├── hmc.R                          # HMC implementation utilities
│   └── advi.R                         # ADVI implementation utilities
│
├── solvers/
│   ├── exponential_exact.R            # Exact solution for exp growth
│   ├── exponential_numeric.R          # Numeric solver for exp growth
│   ├── heat_exact.R                   # Exact solution for heat equation
│   ├── heat_numeric.R                 # Numeric solver for heat equation
│   └── adr_numeric.R                  # Numeric solver for ADR equation
│
├── case_studies/
│   ├── exponential_growth/
│   │   ├── exp_regression.R           # Non-linear regression approach
│   │   └── exp_gp.R                   # GP approach
│   │
│   ├── heat_equation/
│   │   ├── heat_regression.R          # Non-linear regression approach
│   │   └── heat_gp.R                  # GP approach
│   │
│   └── advection_diffusion_reaction/
│       └── adr_regression.R           # Non-linear regression approach
│
└── experiments/
    ├── comparison_hmc_advi.R          # Comparison of inference methods
    ├── comparison_exact_numeric.R     # Comparison of solution approaches
    └── predictive_performance.R       # Analysis of predictive performance

## Case Studies

We demonstrate our framework through three case studies of increasing complexity:

- **Exponential Growth Model**: A simple one-dimensional ODE
- **Heat Equation**: A two-dimensional spatiotemporal PDE
- **Advection-Diffusion-Reaction System**: A complex PDE with six unknown parameters

## Implementation

All code is implemented in R using:
- Stan for Bayesian inference
- ggplot2 with a publication-quality theme for visualizations

## Repository Structure

- `/code`: R and Stan implementation files
- `/figures`: Generated visualizations
- `/data`: Simulation data for case studies
- `/paper`: Supplementary materials related to the paper

## Getting Started

```r
# Install required packages
install.packages(c("rstan", "ggplot2", "reshape2", "dplyr", "bayesplot", "patchwork"))

# Clone the repository
# Run the examples in the /code directory
