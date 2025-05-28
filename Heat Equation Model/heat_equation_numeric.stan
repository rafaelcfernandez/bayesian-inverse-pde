// heat_equation_numeric.stan
// Bayesian inverse problem for heat conduction equation
// du/dt - alpha * d²u/dx² = 0, with Gaussian initial condition
// Numerical solution using finite differences method

data {
  int<lower=1> n;                    // Number of spatial points
  int<lower=1> nt;                   // Number of time steps
  matrix[n, nt] f_obs;               // Observed temperature matrix
}

parameters {
  real<lower=0, upper=0.5> w;        // Numerical stability parameter
  real<lower=0> sigma_y;             // Measurement error standard deviation
}

model {
  matrix[n, nt] f;                   // Calculated temperature values

  // Initialize with observed initial condition
  f[, 1] = f_obs[, 1]; 

  // Finite differences scheme
  for (t in 1:(nt - 1)) {
    for (x in 2:(n - 1)) {
      f[x, t+1] = (1 - 2 * w) * f[x, t] + w * (f[x+1, t] + f[x-1, t]);
    }
    
    // Boundary conditions
    f[1, t+1] = 0;                   // Left boundary
    f[n, t+1] = 0;                   // Right boundary
  }

  // Likelihood: compare calculated f with observed data
  for (t in 2:nt) {
    f_obs[, t] ~ normal(f[, t], sigma_y);
  }
  
  // Priors
  w ~ beta(2, 6);                    // Stability parameter prior
  sigma_y ~ cauchy(0, 0.1);          // Measurement error prior
}

