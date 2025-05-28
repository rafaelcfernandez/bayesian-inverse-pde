// gaussian_process_linear.stan
// Gaussian Process with linear mean function
// GP regression with linear trend: mu(t) = beta_0 + beta_1*t
// Used for comparison against physics-informed GP

functions {
  // RBF kernel: k(t1,t2) = sigma2 * exp(-0.5 * (t1-t2)²/length_scale²)
  matrix sq_exp_kernel(vector t, real sigma2, real length_scale) {
    int n = rows(t);
    matrix[n, n] K;
    
    for (i in 1:n) {
      for (j in i:n) {
        real dist_sq = square(t[i] - t[j]);
        K[i, j] = sigma2 * exp(-0.5 * dist_sq / square(length_scale));
        if (i != j) {
          K[j, i] = K[i, j];          // Symmetry
        }
      }
    }
    return K;
  }
}

data {
  int<lower=1> n_obs;                  // Number of observed points
  vector[n_obs] t_obs;                 // Time points for observations
  vector[n_obs] y_obs;                 // Observations
  matrix[n_obs, 2] X;                  // Design matrix [1, t]
  real<lower=0> b1;                    // Parameter for length_scale prior
  real<lower=0> b2;                    // Parameter for sigma2_kernel prior
}

parameters {
  vector[2] beta;                      // Linear coefficients [beta_0, beta_1]
  real<lower=0> sigma_y;               // Observation noise standard deviation
  real<lower=0> sigma2_kernel;         // Gaussian kernel variance
  real<lower=0> length_scale;          // Kernel length scale
}

model {
  // Priors for linear coefficients
  beta ~ normal(0, 10);                // Vague prior for coefficients
  
  // Priors for other parameters
  sigma_y ~ cauchy(0, 10);             // Observation noise prior
  sigma2_kernel ~ inv_gamma(2, b2);    // Kernel variance prior
  length_scale ~ inv_gamma(2, b1);     // Length scale prior
  
  {
    // Deterministic part (linear mean)
    vector[n_obs] mu = X * beta;
    
    // Stochastic part (Gaussian process)
    matrix[n_obs, n_obs] K = sq_exp_kernel(t_obs, sigma2_kernel, length_scale);
    
    // Add observation noise
    for (i in 1:n_obs) {
      K[i, i] = K[i, i] + square(sigma_y);
    }
    
    // Joint likelihood (GP with non-zero mean)
    y_obs ~ multi_normal(mu, K);
  }
}
