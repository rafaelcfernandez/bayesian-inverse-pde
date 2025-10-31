# HEADER ---------------------------------------------------------------------------------------
# Burgers Equation - Bayesian Parameter Estimation via B-PIGP
# Inverse problem using Cole-Hopf transformation and Gaussian Process regression

rm(list = ls())  # Clear environment
gc()             # Clear memory

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
  rstan,       # Bayesian models via Stan
  ggplot2,     # Visualization
  reshape2,    # For melt function
  viridis,     # Color palettes
  plotly       # Interactive plots
)

cat("\014")  # Clear console

# DATA GENERATION -----------------------------------------------------------------------------

set.seed(1234)  # For reproducibility

# Domain parameters
L           <- 10.0            # Domain length
T_time      <- 1.0             # Total simulation time
Nx          <- 35              # Number of spatial points
Nt          <- 50              # Number of time points

# Physical parameters (true values)
alpha_true  <- 0.1             # Viscosity
sigma_c     <- 1.0             # Initial condition spread
sigma_e     <- 0.01            # Heat equation noise
sigma_y     <- 0.01            # Measurement noise
valor_borda <- 1e-3            # Minimum value to avoid log(0)

# Grid setup
x <- seq(0, L, length.out = Nx)
t <- seq(0, T_time, length.out = Nt)

# Heat equation solution via Fourier series
n_termos <- 200
A_n <- numeric(n_termos)

for (n in 1:n_termos) {
  x_int          <- seq(0, L, length.out = 1001)
  dx             <- x_int[2] - x_int[1]
  integrand_vals <- exp(-((x_int - L/2)^2) / (2 * sigma_c^2)) * sin(n * pi * x_int / L)
  A_n[n]         <- (2/L) * dx * (sum(integrand_vals) - 
                                    0.5 * (integrand_vals[1] + integrand_vals[length(integrand_vals)]))
}

u_heat <- matrix(0, Nx, Nt)

for (i in 1:Nx) {
  for (j in 1:Nt) {
    soma <- 0
    for (n in 1:n_termos) {
      lambda_n <- (n * pi / L)^2
      termo    <- A_n[n] * sin(n * pi * x[i] / L) * exp(-lambda_n * alpha_true * t[j])
      soma     <- soma + termo
    }
    u_heat[i, j] <- soma
  }
}

u_heat[u_heat < valor_borda] <- valor_borda

# Add noise to heat equation
u_heat_obs <- u_heat + matrix(rnorm(Nx * Nt, mean = 0, sd = sigma_e), nrow = Nx, ncol = Nt)
u_heat_obs[u_heat_obs < valor_borda] <- valor_borda

# Cole-Hopf transformation to Burgers equation
dx <- x[2] - x[1]
v_burgers <- matrix(0, Nx, Nt)

for (j in 1:Nt) {
  ln_u <- log(u_heat_obs[, j])
  
  # 4th order spatial derivative
  for (i in 3:(Nx - 2)) {
    dln_u_dx    <- (-ln_u[i + 2] + 8 * ln_u[i + 1] - 8 * ln_u[i - 1] + ln_u[i - 2]) / (12 * dx)
    v_burgers[i, j] <- -2 * alpha_true * dln_u_dx
  }
  
  # Boundaries
  v_burgers[1, j]      <- -2 * alpha_true * (ln_u[2] - ln_u[1]) / dx
  v_burgers[2, j]      <- -2 * alpha_true * (ln_u[3] - ln_u[1]) / (2 * dx)
  v_burgers[Nx - 1, j] <- -2 * alpha_true * (ln_u[Nx] - ln_u[Nx - 2]) / (2 * dx)
  v_burgers[Nx, j]     <- -2 * alpha_true * (ln_u[Nx] - ln_u[Nx - 1]) / dx
}

# Add observation noise to Burgers solution
v_obs <- v_burgers + matrix(rnorm(Nx * Nt, mean = 0, sd = sigma_y), nrow = Nx, ncol = Nt)

# Prepare data for visualization
data <- expand.grid(x = x, t = t)
data$u_heat_true <- as.vector(u_heat)
data$u_heat_obs  <- as.vector(u_heat_obs)
data$v_true      <- as.vector(v_burgers)
data$v_obs       <- as.vector(v_obs)

# Heatmap visualizations
plot_ly(data, x = ~x, y = ~t, z = ~u_heat_true, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Heat Equation - True Solution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)'))

plot_ly(data, x = ~x, y = ~t, z = ~v_true, type = 'contour', colorscale = 'Plasma') %>%
  layout(title = 'Burgers Equation - True Solution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)'))

plot_ly(data, x = ~x, y = ~t, z = ~v_obs, type = 'contour', colorscale = 'Plasma') %>%
  layout(title = 'Burgers Equation - Observed Solution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)'))

# Sampling strategy: separate boundary and interior points
pct_sample <- 0.30
total_points <- Nx * Nt
n_total <- round(total_points * pct_sample)

ratio <- Nx / Nt
n_points_t <- round(sqrt(n_total / ratio))
n_points_x <- round(n_total / n_points_t)

n_points_x <- max(3, min(n_points_x, Nx))
n_points_t <- max(3, min(n_points_t, Nt))

x_indices <- round(seq(1, Nx, length.out = n_points_x))
t_indices <- round(seq(1, Nt, length.out = n_points_t))

sampled_coords <- expand.grid(i = x_indices, j = t_indices)
sampled_coords$x <- x[sampled_coords$i]
sampled_coords$t <- t[sampled_coords$j]
sampled_coords$v <- v_obs[cbind(sampled_coords$i, sampled_coords$j)]

# Classify: boundary (i=1, i=Nx, or j=1) vs interior
sampled_coords$type <- ifelse(
  sampled_coords$i == 1 | sampled_coords$i == Nx | sampled_coords$j == 1,
  "Boundary", "Interior"
)

boundary_points <- subset(sampled_coords, type == "Boundary")
interior_points <- subset(sampled_coords, type == "Interior")

x_0 <- boundary_points$x
t_0 <- boundary_points$t
b_0 <- boundary_points$v

x_d <- interior_points$x
t_d <- interior_points$t
y_d <- interior_points$v

# Prior parameters
alpha_gamma <- 0.01
grid <- seq(0, 200, length.out = 100000)

# Helper function for inverse gamma quantile matching
beta_ig <- function(arg) {
  u_candidate <- pinvgamma(arg, 2, grid)
  u_diff <- abs((1 - alpha_gamma) - u_candidate)
  grid[which.min(u_diff)]
}

b_rx     <- beta_ig(1.0)
b_rt     <- beta_ig(0.1)
b_sigma2 <- beta_ig(1.0)

# STAN MODEL ----------------------------------------------------------------------------------

model <- stan_model(file = "burgers.stan")

# Data preparation for Stan
data_list <- list(
  n_0           = length(x_0),
  n_d           = length(x_d),
  x_0           = x_0,
  t_0           = t_0,
  b_0           = b_0,
  x_d           = x_d,
  t_d           = t_d,
  y_d           = y_d,
  b_rx          = b_rx,
  b_rt          = b_rt,
  b_sigma2      = b_sigma2,
  alpha_mean    = 0.0,
  alpha_sd      = 1.0,
  sigma_y_mean  = sigma_y,
  sigma_y_scale = 0.1,
  jitter        = 1e-6
)

# FITTING -------------------------------------------------------------------------------------

fit_hmc <- sampling(model, data = data_list, iter = 2000, warmup = 1000, seed = 123, chains = 1)