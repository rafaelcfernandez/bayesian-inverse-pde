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
├── inference/
│   ├── stan_models/
│   │   ├── nonlinear_regression.stan
│   │   └── gaussian_process.stan
│   ├── hmc.R
│   └── advi.R
│
├── solvers/
│   ├── exponential_exact.R
│   ├── exponential_numeric.R
│   ├── heat_exact.R
│   ├── heat_numeric.R
│   └── adr_numeric.R
│
├── case_studies/
│   ├── exponential_growth/
│   │   ├── exp_regression.R
│   │   └── exp_gp.R
│   ├── heat_equation/
│   │   ├── heat_regression.R
│   │   └── heat_gp.R
│   └── advection_diffusion_reaction/
│       └── adr_regression.R
│
└── experiments/
├── comparison_hmc_advi.R
├── comparison_exact_numeric.R
└── predictive_performance.R

This structure organizes the code into four main components:
- **inference**: Stan models and inference method utilities
- **solvers**: Exact and numerical solvers for each equation
- **case_studies**: Implementation of both approaches for each example
- **experiments**: Comparative analyses across methods

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
