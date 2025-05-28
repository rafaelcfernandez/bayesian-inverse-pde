# HEADER ---------------------------------------------------------------------------------------
# Exponential Growth Model - Analytical Solution
# Bayesian parameter estimation for exponential growth equation using exact solution

rm(list=ls()) # Clear environment
gc() # Clear memory
require(rstan) 
require(ggplot2)
require(plotly)
cat("\014") # Clear console

# DATA GENERATION -----------------------------------------------------------------------------

set.seed(1234) # Set random seed for reproducibility
n       <- 8                           # Sample size
t       <- seq(0, 2, length.out = n)   # Time points
lambda  <- 1                           # Growth rate parameter
sigma_y <- 0.1                         # Measurement error standard deviation
eps     <- rnorm(n, 0, sigma_y)        # Generate measurement error
y       <- exp(lambda * t) + eps       # Generate response variable
data    <- list(n = n, t = t, y = y)   # Data list for Stan
df      <- data.frame(t = t, y = y)     # Data frame for plotting

plot(df$y)

# STAN MODEL ----------------------------------------------------------------------------------

model <- stan_model(file = "exponential_growth_regression.stan")

# FITTING -------------------------------------------------------------------------------------

fit_hmc <- sampling(model, data = data, iter = 2000, warmup = 1000, seed = 123, chains = 1)