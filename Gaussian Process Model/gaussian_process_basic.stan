// gaussian_process_basic.stan
// Basic Gaussian Process without physical constraints
// Standard GP regression for exponential growth data
// Used as baseline comparison against physics-informed GP

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
  real<lower=0> b1;                    // Parameter for length_scale prior
  real<lower=0> b2;                    // Parameter for sigma2_kernel prior
}

parameters {
  real<lower=0> sigma_y;               // Observation noise standard deviation
  real<lower=0> sigma2_kernel;         // Gaussian kernel variance
  real<lower=0> length_scale;          // Kernel length scale
  vector[n_obs] f;                     // Latent GP function values
}

model {
  // Priors
  sigma_y ~ cauchy(0, 10);             // Observation noise prior
  sigma2_kernel ~ inv_gamma(2, b2);    // Kernel variance prior
  length_scale ~ inv_gamma(2, b1);     // Length scale prior
  
  // Gaussian Process covariance
  {
    matrix[n_obs, n_obs] K = sq_exp_kernel(t_obs, sigma2_kernel, length_scale);
    
    // Add small jitter for numerical stability
    for (i in 1:n_obs) {
      K[i, i] = K[i, i] + 1e-8;
    }
    
    // Prior for latent GP values
    f ~ multi_normal(rep_vector(0, n_obs), K);
  }
  
  // Likelihood
  y_obs ~ normal(f, sigma_y);
}

