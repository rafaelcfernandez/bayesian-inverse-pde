// gaussian_process_physics.stan
// Bayesian Physics-Informed Gaussian Process (b-PIGP)
// GP with physical constraints from exponential growth ODE: du/dt - lambda*u = 0
// Implements conditioning on differential equation constraint

functions {
  // Squared exponential kernel: k(t1,t2) = sigma2 * exp(-0.5 * (t1-t2)²/length_scale²)
  real sq_exp_kernel(real t1, real t2, real sigma2, real length_scale) {
    real dist_sq = square(t1 - t2);
    return sigma2 * exp(-0.5 * dist_sq / square(length_scale));
  }
  
  // Kernel derivative with respect to t2: ∂k/∂t2
  real deriv_kernel_t2(real t1, real t2, real sigma2, real length_scale) {
    // Multiply base kernel by derivative term
    real k = sq_exp_kernel(t1, t2, sigma2, length_scale);
    return k * ((t1 - t2) / square(length_scale));
  }
  
  // Kernel derivative with respect to t1: ∂k/∂t1
  real deriv_kernel_t1(real t1, real t2, real sigma2, real length_scale) {
    real k = sq_exp_kernel(t1, t2, sigma2, length_scale);
    return k * (-(t1 - t2) / square(length_scale));
  }
  
  // Second cross derivative of kernel: ∂²k/∂t1∂t2
  real deriv2_kernel_t1_t2(real t1, real t2, real sigma2, real length_scale) {
    real k = sq_exp_kernel(t1, t2, sigma2, length_scale);
    real dist_sq = square(t1 - t2);
    // Term capturing non-linearities in temporal correlation
    return k * (1 - dist_sq / square(length_scale)) / square(length_scale);
  }
  
  // Main function: Build conditional covariance matrix
  // Solves distribution u | f = 0 for constrained Gaussian process
  matrix build_conditional_cov(
    vector t_obs,        // Temporal observation points
    real lambda,         // Differential equation parameter
    real sigma2,         // Kernel variance
    real length_scale,   // Length scale
    real jitter          // Numerical stability
  ) {
    int n_obs = rows(t_obs);
    
    // Initialize block covariance matrices
    // Σ = [Σ_uu  Σ_uf]
    //     [Σ_fu  Σ_ff]
    matrix[n_obs, n_obs] K_uu = rep_matrix(0, n_obs, n_obs);
    matrix[n_obs, n_obs] K_uf = rep_matrix(0, n_obs, n_obs);
    matrix[n_obs, n_obs] K_fu = rep_matrix(0, n_obs, n_obs);
    matrix[n_obs, n_obs] K_ff = rep_matrix(0, n_obs, n_obs);
    
    // Block K_uu: Covariance between function values u(t)
    // Uses squared exponential kernel to measure correlation
    for (i in 1:n_obs) {
      for (j in 1:n_obs) {
        K_uu[i, j] = sq_exp_kernel(t_obs[i], t_obs[j], sigma2, length_scale);
      }
    }
    
    // Block K_uf: Covariance between u(t) and f(t) = du/dt - λu(t)
    // Incorporates differential equation constraint
    for (i in 1:n_obs) {
      for (j in 1:n_obs) {
        real d_dt2 = deriv_kernel_t2(t_obs[i], t_obs[j], sigma2, length_scale);
        real k_base = sq_exp_kernel(t_obs[i], t_obs[j], sigma2, length_scale);
        K_uf[i, j] = d_dt2 - lambda * k_base;
      }
    }
    
    // Block K_fu: Transpose of K_uf by symmetry
    K_fu = K_uf';
    
    // Block K_ff: Covariance between derivatives f(t)
    // Captures complex dynamics of differential equation
    for (i in 1:n_obs) {
      for (j in 1:n_obs) {
        real d2 = deriv2_kernel_t1_t2(t_obs[i], t_obs[j], sigma2, length_scale);
        real d_dt1 = deriv_kernel_t1(t_obs[i], t_obs[j], sigma2, length_scale);
        real d_dt2 = deriv_kernel_t2(t_obs[i], t_obs[j], sigma2, length_scale);
        real k_base = sq_exp_kernel(t_obs[i], t_obs[j], sigma2, length_scale);
        
        // Complex term incorporating ODE dynamics
        K_ff[i, j] = d2 - lambda * d_dt1 - lambda * d_dt2 + square(lambda) * k_base;
      }
      // Add jitter for numerical stability
      K_ff[i, i] = K_ff[i, i] + jitter;
    }
    
    {
      // STEP 1: Calculate K_ff^(-1) * K_fu
      // Corresponds to: Σ_FF^(-1) * Σ_FU
      // Function: Transform constraint f(t) = 0 into information about u(t)
      matrix[n_obs, n_obs] K_ff_inv_K_fu = mdivide_left_spd(K_ff, K_fu);

      // STEP 2: Calculate conditional covariance
      // Exact mathematical formula: Σ_u|f = Σ_uu - Σ_uf * Σ_ff^(-1) * Σ_fu
      // Function: Reduce original covariance incorporating constraint f(t) = 0
      matrix[n_obs, n_obs] Sigma_u_given_f = K_uu - K_uf * K_ff_inv_K_fu;
      
      // Ensure covariance matrix symmetry
      Sigma_u_given_f = 0.5 * (Sigma_u_given_f + Sigma_u_given_f');
      
      return Sigma_u_given_f;
    }
  }
}

data {
  int<lower=1> n_obs;                  // Number of observed points
  vector[n_obs] t_obs;                 // Time points for observations
  vector[n_obs] y_u;                   // Observations of u(t)
  real<lower=0> b1;                    // Parameter for length_scale prior
  real<lower=0> b2;                    // Parameter for sigma2_kernel prior
}

parameters {
  real<lower=0> lambda;                // ODE parameter (growth rate)
  real<lower=0> sigma_y;               // Observation noise standard deviation
  real<lower=0> sigma2_kernel;         // Gaussian kernel variance
  real<lower=0> length_scale;          // Kernel length scale
}

model {
  // Priors with physical interpretation
  // lambda: growth rate with normal distribution centered at 0
  lambda ~ normal(0, 10);              // Growth rate parameter prior
  
  // sigma_y: observation noise with robust Cauchy distribution
  sigma_y ~ cauchy(0, 10);             // Observation noise prior
  
  // Priors for kernel hyperparameters using Inverse Gamma
  sigma2_kernel ~ inv_gamma(2, b2);    // Kernel variance prior
  length_scale ~ inv_gamma(2, b1);     // Length scale prior
  
  {
    // Calculate conditional covariance matrix u | f = 0
    matrix[n_obs, n_obs] Sigma_u_given_f = build_conditional_cov(
      t_obs, lambda, sigma2_kernel, length_scale, 1e-8
    );
    
    // Add observation noise to covariance diagonal
    matrix[n_obs, n_obs] Sigma_obs = Sigma_u_given_f;
    for (i in 1:n_obs) {
      Sigma_obs[i, i] = Sigma_obs[i, i] + square(sigma_y);
    }
    
    // Likelihood: multivariate normal distribution
    // Zero mean, as GP has zero mean and f = 0 doesn't change this
    y_u ~ multi_normal(rep_vector(0, n_obs), Sigma_obs);
  }
}

