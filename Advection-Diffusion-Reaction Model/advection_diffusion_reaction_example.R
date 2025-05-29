# HEADER ---------------------------------------------------------------------------------------
# Advection-Diffusion-Reaction Equation - Bayesian Parameter Estimation
# Inverse problem using finite differences approximation

rm(list=ls()) # Clear environment
gc() # Clear memory

# Global settings
options(
  mc.cores = parallel::detectCores(),  # Parallel processing
  scipen = 999,                        # Disable scientific notation
  digits = 15,                         # Increase numerical precision (max 22)
  stringsAsFactors = FALSE,            # Avoid automatic factor conversion
  width = 120                          # Console output width
)

# Load required packages
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  rstan,    # Bayesian models via Stan
  ggplot2,  # Visualization
  reshape2, # For melt function
  viridis,  # Color palettes
  plotly    # Interactive plots
)

cat("\014") # Clear console

# DATA GENERATION -----------------------------------------------------------------------------

set.seed(123)  # For reproducibility

# Domain parameters
L         <- 20.0       # Domain length
T_time    <- 15.0       # Total simulation time
Nx        <- 50         # Number of spatial points
Nt        <- 100        # Number of time points

# Physical parameters (true values)
D_true        <- 0.06       # Diffusion coefficient
V_true        <- 0.7        # Advection velocity
R_true        <- 0.3        # Reaction coefficient
lambda_l_true <- 7.8        # Source location
lambda_t_true <- 5.3        # Source time
lambda_i_true <- 1.0        # Source intensity
sigma_x_true  <- sqrt(0.5)  # Spatial source spread (FIXED)
sigma_t_true  <- sqrt(0.5)  # Temporal source spread (FIXED)
sigma_y       <- 0.03       # Measurement error

# Grid setup
dx <- L / (Nx - 1)
dt <- T_time / (Nt - 1)
x <- seq(0, L, length.out = Nx)
t <- seq(0, T_time, length.out = Nt)

# Initialize concentration matrix
u <- matrix(0, nrow = Nx, ncol = Nt)

# Initial condition (all zeros)
u[, 1] <- 0

# Finite differences solution
for (n in 1:(Nt - 1)) {
  for (i in 2:(Nx - 1)) {
    # Diffusion term
    diffusion <- D_true * (u[i+1, n] - 2 * u[i, n] + u[i-1, n]) / dx^2
    
    # Advection term
    advection <- -V_true * (u[i+1, n] - u[i-1, n]) / (2 * dx)
    
    # Reaction term
    reaction <- -R_true * u[i, n]
    
    # Source term
    x_term <- -((x[i] - lambda_l_true)^2) / (2 * sigma_x_true^2)
    t_term <- -((t[n] - lambda_t_true)^2) / (2 * sigma_t_true^2)
    sigmoid_term <- 1 / (1 + exp(-100 * (t[n] - lambda_t_true)))
    source <- lambda_i_true * exp(x_term + t_term) * sigmoid_term
    
    # Update concentration
    u[i, n+1] <- u[i, n] + dt * (diffusion + advection + reaction + source)
    
    # Ensure non-negativity
    if (u[i, n+1] < 0) u[i, n+1] <- 0
  }
  
  # Boundary conditions (Neumann)
  u[1, n+1] <- u[2, n+1]        # Left boundary
  u[Nx, n+1] <- u[Nx-1, n+1]   # Right boundary
}

# Add observation noise
u_obs <- u + matrix(rnorm(Nx * Nt, mean = 0, sd = sigma_y), nrow = Nx, ncol = Nt)
u_obs[u_obs < 0] <- 0  # Ensure non-negativity

# Prepare data for visualization
data <- expand.grid(x = x, t = t)
data$concentration_true <- as.vector(u)
data$concentration_obs <- as.vector(u_obs)

# Concentration evolution visualization
ggplot(data, aes(x = x, y = concentration_true, color = factor(t))) +
  geom_line(linewidth = 1) +
  theme_bw() +
  theme(legend.position = "none") +
  labs(title = "Concentration Distribution Along Position and Time", 
       x = "Position (x)", 
       y = "Concentration", 
       color = "Time (t)") -> p1

# Heatmap visualizations
plot_ly(data, x = ~x, y = ~t, z = ~concentration_true, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'True Concentration Distribution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)')) -> p2

plot_ly(data, x = ~x, y = ~t, z = ~concentration_obs, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Observed Concentration Distribution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)')) -> p3

# Side-by-side comparison
subplot(p2, p3, nrows = 1, shareX = TRUE, shareY = TRUE) %>%
  layout(title = 'Comparison: True vs. Observed Concentration')

# STAN MODEL ----------------------------------------------------------------------------------

model <- stan_model(file = "advection_diffusion_reaction_regression.stan")

# Data preparation for Stan
data_list <- list(
  Nx = Nx,
  Nt = Nt,
  delta_t = dt,
  delta_x = dx,
  L = L,
  T = T_time,
  u_obs = u_obs,
  sigma_x = sigma_x_true,  # Pass as fixed data
  sigma_t = sigma_t_true   # Pass as fixed data
)

# FITTING -------------------------------------------------------------------------------------

fit_hmc <- sampling(model, data = data_list, iter = 2000, warmup = 1000, seed = 123, chains = 1)