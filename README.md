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

### Exponential Growth Model
- `exponential_growth_regression.stan` - Non-linear regression for exact analytical solution
- `exponential_growth_numeric.stan` - Non-linear regression with finite differences approximation
- `exponential_growth_example.R` - Complete implementation with synthetic data generation

### Heat Equation
- `heat_equation_regression.stan` - Non-linear regression for exact solution
- `heat_equation_numeric.stan` - Non-linear regression with finite differences
- `heat_equation_example.R` - Complete implementation with synthetic data generation

### Advection-Diffusion-Reaction System
- `advection_diffusion_reaction_regression.stan` - Non-linear regression with finite differences
- `advection_diffusion_reaction_example.R` - Complete implementation with synthetic data generation
- `advection_diffusion_reaction_variational.stan` - ADVI implementation

### Gaussian Process Models
- `gaussian_process_basic.stan` - Basic Gaussian Process without physical constraints
- `gaussian_process_basic_example.R` - Complete implementation with synthetic data generation
- `gaussian_process_linear.stan` - Gaussian Process with linear mean function
- `gaussian_process_quadratic.stan` - Gaussian Process with quadratic mean function
- `gaussian_process_physics.stan` - Gaussian Process with physical constraints (b-PIGP)

## Case Studies

We demonstrate our framework through three case studies of increasing complexity:

- **Exponential Growth Model**: A simple one-dimensional ODE
- **Heat Equation**: A two-dimensional spatiotemporal PDE  
- **Advection-Diffusion-Reaction System**: A complex PDE with six unknown parameters

## Implementation

All code is implemented in R using:
- Stan for Bayesian inference
- ggplot2 with publication-quality themes for visualizations
- Standardized code structure across all examples
- Self-contained synthetic data generation in each example

## Getting Started

```r
# Install required packages
install.packages(c("rstan", "ggplot2", "reshape2", "viridis", "plotly"))

# Run example implementations
source("advection_diffusion_reaction_example.R")  # ADR system
source("gaussian_process_basic_example.R")        # Basic GP model