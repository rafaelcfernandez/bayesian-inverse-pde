# HEADER ---------------------------------------------------------------------------------------
# HEAT EQUATION WITH ANALYTICAL SOLUTION

rm(list=ls()) # remover todos os itens do ambiente
gc() # limpar a memória

# Configurações globais
options(
  mc.cores = parallel::detectCores(),  # paralelização 
  scipen = 999,                        # desabilitar notação científica
  digits = 15,                         # aumentar precisão numérica (máximo 22)
  stringsAsFactors = FALSE,            # evitar conversão automática para fatores
  width = 120                           # largura da saída no console
)

# Carregar pacotes necessários
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  rstan,    # modelos Bayesianos via Stan
  ggplot2,  # visualização
  reshape2, # para melt
  viridis,  # paletas de cores
  plotly    # gráficos interativos
)

cat("\014") # limpar o console


# DATA ---------------------------------------------------------------------------------------

set.seed(123)  # Para reprodutibilidade

L       <- 10       # comprimento da barra
sigma_c <- 1        # largura da gaussiana
alpha   <- 1        # coeficiente de difusão
Nx      <- 35       # Número de pontos no espaço
Nt      <- 50        # Número de pontos no tempo
x       <- seq(0, L, length.out = Nx) # discretização no espaço
t       <- seq(0, 1, length.out = Nt) # discretização no tempo
sigma_e <- 0.01     # erro de medição

# Solução gaussiana
y <- matrix(0, nrow = Nx, ncol = Nt)
for (n in 1:Nt) {
  for (i in 1:Nx) {
    y[i, n] <- exp(-((x[i] - L/2)^2) / (2 * sigma_c^2)) * exp(-alpha * t[n])
  }
}

y_expe           <- y + rnorm(Nx * Nt, mean = 0, sd = sigma_e)  # Dados experimentais com ruído
y_expe[y_expe<0] <- 0
data             <- expand.grid(x = x, t = t)
data$y_real      <- as.vector(y)
data$y_expe      <- as.vector(y_expe)

# Visualizações em heatmap
plot_ly(data, x = ~x, y = ~t, z = ~y_real, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Distribuição de Temperatura Real',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p2

plot_ly(data, x = ~x, y = ~t, z = ~y_expe, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Distribuição de Temperatura Observada',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p3

# Comparação lado a lado
subplot(p2, p3, nrows = 1, shareX = TRUE, shareY = TRUE)  


# MODEL ---------------------------------------------------------------------------------------

# Preparação dos dados para Stan
data_list <- list(
  N = length(data$y_expe),
  x = data$x,
  t = data$t,
  y = c(data$y_expe),
  L = L
)

stan_model_code <- 
  
  "
data {
  int<lower=1> N;         // Número de pontos de dados
  vector[N] x;            // Posições no espaço
  vector[N] t;            // Tempos
  vector[N] y;            // Valores observados (f(x,t))
  real<lower=0> L;        // Comprimento da barra
}

parameters {
  real<lower=0> sigma_c;  // Parâmetro sigma_c (largura da gaussiana)
  real<lower=0> alpha;    // Coeficiente de difusão
  real<lower=0> sigma_e;  // Desvio padrão do ruído
}

model {
  vector[N] mu;           // Valores esperados pelo modelo
  
  // Definindo a solução esperada pela equação do calor
  for (n in 1:N) {
    mu[n] = exp(-(pow(x[n] - L/2, 2)) / (2 * pow(sigma_c, 2))) * exp(-alpha * t[n]);
  }
  
  // Verossimilhança com ruído aditivo gaussiano
  y ~ normal(mu, sigma_e);  // Distribuição normal para as observações com ruído
  
  // Prioris
  sigma_c ~ normal(1, 0.5);   // Prior para sigma_c
  alpha ~ normal(1, 0.5);     // Prior para alpha
  sigma_e ~ cauchy(0, 0.1);   // Prior para o desvio padrao do ruido
}
"

# Compilar modelo Stan
model <- stan_model(model_code = stan_model_code)


# FITTING ---------------------------------------------------------------------------------------

fit_hmc  <- sampling(model, data=data_list, iter=2000, warmup=1000, seed=1, chains=1)  


# SAVING ---------------------------------------------------------------------------------------

saveRDS(fit_hmc,"Objetos/Regressao/fit_heat_analitico_hmc.rds")


# READING ---------------------------------------------------------------------------------------

fit_hmc  <- readRDS("Objetos/Regressao/fit_heat_analitico_hmc.rds")


# PLOTS ---------------------------------------------------------------------------------------

fit1 <- rstan::extract(fit_hmc)

# Plotando os parâmetros - sem comparação com ADVI
par(mfrow=c(2,2))
plot(fit1$alpha, type="l", xlab = "Iteration", ylab = expression(alpha), main="Traço do parâmetro alpha") 
abline(h=alpha, lty=2, lwd=2, col="red")

hist(fit1$alpha, ylab="Densidade", xlab = expression(alpha), 
     prob = TRUE, main="Distribuição posterior de alpha", border=FALSE, col="skyblue")
abline(v=alpha, lty=2, lwd=2, col="red")

plot(fit1$sigma_c, type="l", xlab = "Iteration", ylab = expression(sigma[c]), main="Traço do parâmetro sigma_c") 
abline(h=sigma_c, lty=2, lwd=2, col="red")

hist(fit1$sigma_c, ylab="Densidade", xlab = expression(sigma[c]), 
     prob = TRUE, main="Distribuição posterior de sigma_c", border=FALSE, col="skyblue")
abline(v=sigma_c, lty=2, lwd=2, col="red")

par(mfrow=c(1,2))
plot(fit1$sigma_e, type="l", xlab = "Iteration", ylab = expression(sigma[e]), main="Traço do parâmetro sigma_e") 
abline(h=sigma_e, lty=2, lwd=2, col="red")

hist(fit1$sigma_e, ylab="Densidade", xlab = expression(sigma[e]), 
     prob = TRUE, main="Distribuição posterior de sigma_e", border=FALSE, col="skyblue")
abline(v=sigma_e, lty=2, lwd=2, col="red")


# ESTIMATED SURFACE ---------------------------------------------------------------------------------------

fit_hmc  <- readRDS("Objetos/Regressao/fit_heat_analitico_hmc.rds")
fit1 <- rstan::extract(fit_hmc)

# Parâmetros estimados
L            <- 10
sigma_c_pred <- median(fit1$sigma_c)
alpha_pred   <- median(fit1$alpha)

# Gerando superfície predita com parâmetros estimados
y_pred <- matrix(0, nrow = Nx, ncol = Nt)
for (n in 1:Nt) {
  for (i in 1:Nx) {
    y_pred[i, n] <- exp(-((x[i] - L/2)^2) / (2 * sigma_c_pred^2)) * exp(-alpha_pred * t[n])
  }
}

# Preparação dos dados para visualização
data_pred        <- expand.grid(x = x, t = t)
data_pred$y_pred <- as.vector(y_pred)

# Plotando superfície predita
plot_ly(data_pred, x = ~x, y = ~t, z = ~y_pred, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Distribuição de Temperatura Predita (Analítica)',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p4

# Comparações entre superfícies
subplot(p2, p4, nrows = 1, shareX = TRUE, shareY = TRUE)  %>%
  layout(title = 'Comparação: Real vs. Predita')


# END OF CODE ----