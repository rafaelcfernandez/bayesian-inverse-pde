// exponential_growth_regression.stan
// Bayesian inverse problem for exponential growth equation
// du/dt - lambda * u = 0, with u(0) = 1
// Exact analytical solution: u(t) = exp(lambda * t)

data {
  int<lower=0> n;                    // Number of observations
  vector[n] t;                       // Time points
  vector[n] y;                       // Observed responses
}

parameters {
  real<lower=0> lambda;              // Growth rate parameter
  real<lower=0> sigma_y;             // Measurement error standard deviation
}

model {
  // Likelihood: y_i ~ N(u(t_i; lambda), sigma_y^2)
  // where u(t; lambda) = exp(lambda * t)
  for (i in 1:n) {
    y[i] ~ normal(exp(lambda * t[i]), sigma_y);
  }
  
  // Priors
  lambda ~ normal(0, 10);            // Growth rate prior
  sigma_y ~ cauchy(0, 10);           // Measurement error prior
}
