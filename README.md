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

### Advection-Diffusion-Reaction Model
Six-parameter system for environmental contamination modeling and industrial process control.
- `advection_diffusion_reaction_regression.stan` - Non-linear regression with finite differences
- `advection_diffusion_reaction_variational.stan` - ADVI implementation
- `advection_diffusion_reaction_example.R` - Complete implementation with synthetic data

### Gaussian Process Model
Surrogate modeling framework with increasing complexity.
- `gaussian_process_basic.stan` - Basic GP without physical constraints
- `gaussian_process_linear.stan` - GP with linear mean function
- `gaussian_process_quadratic.stan` - GP with quadratic mean function
- `gaussian_process_physics.stan` - Bayesian Physics-Informed GP (b-PIGP)
- `gaussian_process_example.R` - Complete implementation

### Exponential Growth Aluminium Model
Real experimental application: thermal diffusivity estimation in aluminum using 56 temperature measurements.
- `exponential_growth_aluminium.stan` - Physics-informed model
- `aluminum_data.csv` - Experimental temperature data

### Viscous Burger Model
Nonlinear PDE solved via Cole-Hopf transformation.
- `burgers_example.r` - Complete implementation with synthetic data
- `burgers.stan` - Implementation via transformation

## Case Studies

Applications in order of implementation:

1. **Advection-Diffusion-Reaction System**: Six-parameter environmental contamination model
2. **Aluminum Thermal Analysis**: Real experimental data for specific heat capacity estimation
3. **Viscous Burgers Equation**: Nonlinear PDE via Cole-Hopf transformation

## Supplementary Material

Additional tutorial examples and detailed implementations are available in the supplementary material:
- **Exponential Growth Model**: Basic ODE parameter estimation
- **Heat Equation**: Tutorial on spatiotemporal PDEs
- **Extended derivations**: Mathematical details of b-PIGP framework

## Implementation

All code is implemented in R using:
- Stan for Bayesian inference
- ggplot2 with publication-quality themes for visualizations
- Standardized code structure across all examples

## Getting Started
```r
# Install required packages
install.packages(c("rstan", "ggplot2", "reshape2", "viridis", "plotly"))

# Run example implementations
source("Advection-Diffusion-Reaction Model/advection_diffusion_reaction_example.R")
source("Exponential Growth Aluminium Model/aluminum_example.R")
source("Gaussian Process Model/gaussian_process_example.R")
source("Viscous Burger Model/burgers_example.R")
```

## Citation

If you use this code in your research, please cite:
```bibtex
@article{fernandez2025bayesian,
  title={Inverse problems in Differential Equations based models: Bayesian formulations and applications},
  author={Fernandez, R.C. and Zanini, C.T.P. and Schmidt, A.M. and Migon, H.S. and Silva Neto, A.J.},
  journal={Applied Mathematical Modelling},
  year={2025}
}
```

## Acknowledgments

This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001, Fundação Carlos Chagas Filho de Amparo à Pesquisa do Estado do Rio de Janeiro (FAPERJ), Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), and Natural Sciences and Engineering Research Council of Canada Discovery Grant (NSERC-DG).