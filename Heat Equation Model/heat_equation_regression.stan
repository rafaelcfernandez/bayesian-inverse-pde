// heat_equation_regression.stan
// Bayesian inverse problem for heat conduction equation
// du/dt - alpha * d²u/dx² = 0, with Gaussian initial condition
// Exact analytical solution: u(x,t) = exp(-((x-L/2)²)/(2*sigma_c²)) * exp(-alpha*t)

data {
  int<lower=1> n;                    // Number of observations
  vector[n] x;                       // Spatial positions
  vector[n] t;                       // Time points
  vector[n] y;                       // Observed responses
  real<lower=0> L;                   // Bar length
  real<lower=0> sigma_c;             // Gaussian width parameter (fixed)
}

parameters {
  real<lower=0> alpha;               // Thermal diffusivity coefficient
  real<lower=0> sigma_y;             // Measurement error standard deviation
}

model {
  vector[n] mu;                      // Expected values from the model
  
  // Analytical solution of heat equation
  for (i in 1:n) {
    mu[i] = exp(-pow(x[i] - L/2, 2) / (2 * pow(sigma_c, 2))) * exp(-alpha * t[i]);
  }
  
  // Likelihood: y_i ~ N(u(x_i, t_i; alpha), sigma_y²)
  y ~ normal(mu, sigma_y);
  
  // Priors
  alpha ~ normal(1, 0.5);            // Thermal diffusivity prior
  sigma_y ~ cauchy(0, 0.1);          // Measurement error prior
}
