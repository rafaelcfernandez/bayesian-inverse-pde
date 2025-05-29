# HEADER ---------------------------------------------------------------------------------------
# Basic Gaussian Process Regression - Baseline Model
# Standard GP without physical constraints

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
n_obs         <- 20         # Number of observation points
t_min         <- 0.0        # Minimum time
t_max         <- 10.0       # Maximum time

# GP hyperparameters (for data generation)
sigma_y_true       <- 0.2         # Observation noise
sigma2_kernel_true <- 1.5         # Kernel variance
length_scale_true  <- 2.0         # Kernel length scale

# Prior parameters
b1 <- 4.0         # Length scale prior parameter
b2 <- 2.0         # Kernel variance prior parameter

# Time points
t_obs <- sort(runif(n_obs, t_min, t_max))

# Generate synthetic data from GP
# Covariance matrix
K_true <- matrix(0, n_obs, n_obs)
for (i in 1:n_obs) {
  for (j in 1:n_obs) {
    dist_sq <- (t_obs[i] - t_obs[j])^2
    K_true[i, j] <- sigma2_kernel_true * exp(-0.5 * dist_sq / length_scale_true^2)
  }
}

# Add jitter for numerical stability
diag(K_true) <- diag(K_true) + 1e-8

# Sample from GP prior
f_true <- as.vector(mvtnorm::rmvnorm(1, mean = rep(0, n_obs), sigma = K_true))

# Add observation noise
y_obs <- f_true + rnorm(n_obs, mean = 0, sd = sigma_y_true)

# Prepare data for visualization
data <- data.frame(
  time = t_obs,
  f_true = f_true,
  y_observed = y_obs
)

# Function visualization
ggplot(data, aes(x = time)) +
  geom_line(aes(y = f_true), color = "red", linewidth = 1) +
  geom_point(aes(y = y_observed), color = "black", size = 2) +
  theme_bw() +
  labs(title = "GP Generated Data: True Function and Observations", 
       x = "Time (t)", 
       y = "Response (y)") -> p1

# Interactive plot
plot_ly(data, x = ~time, y = ~f_true, type = 'scatter', mode = 'lines', name = 'True Function') %>%
  add_trace(y = ~y_observed, mode = 'markers', name = 'Observations') %>%
  layout(title = 'GP Data: True vs Observed',
         xaxis = list(title = 'Time (t)'),
         yaxis = list(title = 'Response (y)')) -> p2

print(p1)
p2

# STAN MODEL ----------------------------------------------------------------------------------

model <- stan_model(file = "gaussian_process_basic.stan")

# Data preparation for Stan
data_list <- list(
  n_obs = n_obs,
  t_obs = t_obs,
  y_obs = y_obs,
  b1 = b1,
  b2 = b2
)

# FITTING -------------------------------------------------------------------------------------

fit_hmc <- sampling(model, data = data_list, itechainr = 2000, warmup = 1000, seed = 123, chains = 1)