// advection_diffusion_reaction_variational.stan
// Bayesian inverse problem for advection-diffusion-reaction equation
// du/dt + V*du/dx - D*d²u/dx² + R*u = f(x,t), with Gaussian source term
// Numerical solution using finite differences method - optimized for ADVI

data {
  int<lower=1> Nx;                   // Number of spatial points
  int<lower=1> Nt;                   // Number of time steps
  real<lower=0> delta_t;             // Time step size
  real<lower=0> delta_x;             // Spatial step size
  real<lower=0> L;                   // Domain length
  real<lower=0> T;                   // Total time
  matrix[Nx, Nt] u_obs;              // Observed concentration data
  real<lower=0> sigma_x;             // Spatial source spread (fixed)
  real<lower=0> sigma_t;             // Temporal source spread (fixed)
}

parameters {
  real<lower=0.01, upper=0.2> D;     // Diffusion coefficient
  real<lower=0.3, upper=1.2> V;      // Advection velocity
  real<lower=0.1, upper=0.6> R;      // Reaction coefficient
  real<lower=L*0.2, upper=L*0.8> lambda_l;  // Source location
  real<lower=T*0.2, upper=T*0.8> lambda_t;  // Source time
  real<lower=0.5, upper=1.5> lambda_i;      // Source intensity
  real<lower=0.01, upper=0.1> sigma_y;      // Measurement error standard deviation
}

model {
  // Priors
  D ~ normal(0.06, 0.02);            // Diffusion coefficient prior
  V ~ normal(0.7, 0.1);              // Advection velocity prior
  R ~ normal(0.3, 0.05);             // Reaction coefficient prior
  lambda_l ~ normal(L/2, L/4);       // Source location prior
  lambda_t ~ normal(T/2, T/4);       // Source time prior
  lambda_i ~ normal(1.0, 0.2);       // Source intensity prior
  sigma_y ~ normal(0.03, 0.01);      // Measurement error prior
  
  // Likelihood
  {
    matrix[Nx, Nt] u;                // Calculated concentration values
    
    // Initial condition
    for (i in 1:Nx) {
      u[i, 1] = 0;
    }
    
    // Finite differences simulation
    for (n in 1:(Nt - 1)) {
      for (i in 2:(Nx - 1)) {
        real diffusion = D * (u[i+1, n] - 2 * u[i, n] + u[i-1, n]) / (delta_x^2);
        real advection = -V * (u[i+1, n] - u[i-1, n]) / (2 * delta_x);
        real reaction = -R * u[i, n];
        
        // Gaussian source term with simplified calculations
        real x_pos = (i - 1) * delta_x;
        real t_now = (n - 1) * delta_t;
        real x_term = -((x_pos - lambda_l)^2 / (2 * sigma_x^2));
        real t_term = -((t_now - lambda_t)^2 / (2 * sigma_t^2));
        real sigmoid_term = 1 / (1 + exp(-100 * (t_now - lambda_t)));
        real source = lambda_i * exp(x_term + t_term) * sigmoid_term;
        
        u[i, n+1] = u[i, n] + delta_t * (diffusion + advection + source + reaction);
      }
      
      // Boundary conditions
      u[1, n+1] = u[2, n+1];          // Left boundary
      u[Nx, n+1] = u[Nx-1, n+1];      // Right boundary
    }
    
    // Likelihood for significant concentration values
    for (n in 1:Nt) {
      for (i in 1:Nx) {
        if (u_obs[i, n] > 1e-6) {
          u_obs[i, n] ~ normal(u[i, n], sigma_y);
        }
      }
    }
  }
}
