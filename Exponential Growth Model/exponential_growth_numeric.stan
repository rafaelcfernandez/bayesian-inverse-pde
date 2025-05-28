// exponential_growth_numeric.stan
// Bayesian inverse problem for exponential growth equation
// du/dt - lambda * u = 0, with u(0) = 1
// Numerical solution using finite differences approximation

data {
  int<lower=0> n;                    // Number of observations
  real x[n];                         // Time points
  real y[n];                         // Observed responses
  real h;                            // Step size
  real y0;                           // Initial condition
}
parameters {
  real<lower=0> lambda;              // Growth rate parameter
  real<lower=0> sigma_y;             // Measurement error standard deviation
}
model {
  real y_pred[n];                    // Predicted values using finite differences
  y_pred[1] = y0;                    // Initial condition
  for (i in 2:n) {
    y_pred[i] = y_pred[i-1] / (1 - h * lambda); 
  }
  
  // Likelihood
  y ~ normal(y_pred, sigma_y);
  
  // Priors
  lambda ~ normal(0, 10);            // Growth rate prior
  sigma_y ~ cauchy(0, 10);           // Measurement error prior
}
