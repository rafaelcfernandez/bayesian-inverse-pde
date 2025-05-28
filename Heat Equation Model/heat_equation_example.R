# HEADER ---------------------------------------------------------------------------------------
# Heat Equation - Analytical Solution
# Bayesian parameter estimation for heat conduction equation using exact solution

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

L         <- 10       # Bar length
sigma_c   <- 1        # Gaussian width (FIXED VALUE, NOT ESTIMATED)
alpha     <- 1        # Thermal diffusivity coefficient
Nx        <- 35       # Number of spatial points (REDUCED)
Nt        <- 50       # Number of time points
x         <- seq(0, L, length.out = Nx) # Spatial discretization
t         <- seq(0, 1, length.out = Nt) # Time discretization
dt        <- t[2] - t[1]                # Time step
dx        <- x[2] - x[1]                # Spatial step
sigma_y   <- 0.01     # Measurement error

# Analytical solution
y <- matrix(0, nrow = Nx, ncol = Nt)
for (n in 1:Nt) {
  for (i in 1:Nx) {
    y[i, n] <- exp(-((x[i] - L/2)^2) / (2 * sigma_c^2)) * exp(-alpha * t[n])
  }
}

y_obs            <- y + rnorm(Nx * Nt, mean = 0, sd = sigma_y)  # Experimental data with noise
y_obs[y_obs < 0] <- 0
data             <- expand.grid(x = x, t = t)
data$y_true      <- as.vector(y)
data$y_obs       <- as.vector(y_obs)

# Temperature distribution visualization over time
ggplot(data, aes(x = x, y = y_true, color = factor(t))) +
  geom_line(linewidth = 1) +
  theme_bw() +
  theme(legend.position = "none") +
  labs(title = "Temperature Distribution Along Position and Time", 
       x = "Position (x)", 
       y = "Temperature (y)", 
       color = "Time (t)") -> p1

# Heatmap visualizations
plot_ly(data, x = ~x, y = ~t, z = ~y_true, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'True Temperature Distribution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)')) -> p2

plot_ly(data, x = ~x, y = ~t, z = ~y_obs, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Observed Temperature Distribution',
         xaxis = list(title = 'Position (x)'),
         yaxis = list(title = 'Time (t)')) -> p3

# Side-by-side comparison
subplot(p2, p3, nrows = 1, shareX = TRUE, shareY = TRUE) %>%
  layout(title = 'Comparison: Analytical True vs. Observed')

# STAN MODEL ----------------------------------------------------------------------------------

model <- stan_model(file = "heat_equation_regression.stan")

# Data preparation for Stan
data_list <- list(
  n = length(data$y_obs),
  x = data$x,
  t = data$t,
  y = c(data$y_obs),
  L = L,
  sigma_c = sigma_c  # Pass sigma_c as fixed data
)

# FITTING -------------------------------------------------------------------------------------

fit_hmc <- sampling(model, data = data_list, iter = 2000, warmup = 1000, seed = 1, chains = 1)
