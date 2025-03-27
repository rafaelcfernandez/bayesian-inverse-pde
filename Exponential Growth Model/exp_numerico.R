# HEADER ---------------------------------------------------------------------------------------
# HEAT EQUATION: INFERÊNCIA COM DADOS GERADOS PELO SOLVER NUMÉRICO

rm(list=ls()) # remover todos os itens do ambiente
gc() # limpar a memória

# Configurações globais
options(
  mc.cores = parallel::detectCores(),  # paralelização 
  scipen = 999,                        # desabilitar notação científica
  digits = 15,                         # aumentar precisão numérica (máximo 22)
  stringsAsFactors = FALSE,            # evitar conversão automática para fatores
  width = 120                          # largura da saída no console
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


# GERAÇÃO DE DADOS COM SOLVER NUMÉRICO -----------------------------------------

set.seed(123)  # Para reprodutibilidade

# Parâmetros da simulação
L       <- 10        # Comprimento da barra
Nx      <- 35        # Número de pontos no espaço
Nt      <- 50        # Número de pontos no tempo
x       <- seq(0, L, length.out = Nx) # Discretização no espaço
t       <- seq(0, 1, length.out = Nt) # Discretização no tempo
dt      <- t[2] - t[1]                # Passo de tempo
dx      <- x[2] - x[1]                # Passo espacial
alpha   <- 1         # Coeficiente de difusão
sigma_e <- 0.01      # Desvio padrão do ruído

# Calcular valor verdadeiro de w com base em alpha e a discretização
w_true <- alpha * dt / (dx * dx)
cat("Valor verdadeiro de w =", w_true, "\n")

# Verificar se o valor verdadeiro de w satisfaz a condição de estabilidade
if (w_true > 0.5) {
  cat("AVISO: O valor verdadeiro de w =", w_true, "excede 0.5, o que pode causar instabilidade numérica!\n")
  cat("Para este valor de alpha, seria necessário dt <=", 0.5 * dx * dx / alpha, "para garantir estabilidade.\n")
  stop("Abortando devido à instabilidade. Ajuste Nx ou Nt para obter w <= 0.5")
}

# Inicializar com uma gaussiana (condição inicial)
sigma_c <- 1  # Largura da gaussiana para condição inicial
y_numeric <- matrix(0, nrow = Nx, ncol = Nt)
y_numeric[, 1] <- exp(-((x - L/2)^2) / (2 * sigma_c^2))  # Condição inicial gaussiana

# Propagar a solução usando o solver numérico com w_true
for (n in 1:(Nt - 1)) {
  for (i in 2:(Nx - 1)) {
    y_numeric[i, n + 1] <- (1 - 2 * w_true) * y_numeric[i, n] + w_true * (y_numeric[i + 1, n] + y_numeric[i - 1, n])
  }
  
  # Condições de fronteira (Dirichlet)
  y_numeric[1, n + 1]  <- 0
  y_numeric[Nx, n + 1] <- 0
}

# Adicionar ruído
y_obs <- y_numeric + rnorm(Nx * Nt, mean = 0, sd = sigma_e)
y_obs[y_obs < 0] <- 0  # Truncar valores negativos

# VISUALIZAÇÃO DOS DADOS ---------------------------------------------------------

# Formato long para visualização
data <- expand.grid(x = x, t = t)
data$y_true <- as.vector(y_numeric)  # Solução numérica (sem ruído)
data$y_obs <- as.vector(y_obs)       # Solução numérica com ruído

# Visualizar solução numérica sem ruído
plot_ly(data, x = ~x, y = ~t, z = ~y_true, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Distribuição de Temperatura - Solução Numérica Real',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p1

# Visualizar dados observados (com ruído)
plot_ly(data, x = ~x, y = ~t, z = ~y_obs, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Distribuição de Temperatura - Dados Observados',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p2

# Comparar lado a lado
subplot(p1, p2, nrows = 1, shareX = TRUE, shareY = TRUE) %>%
  layout(title = 'Comparação: Numérico Real vs. Observado')


# MODELO DE INFERÊNCIA COM SOLVER NUMÉRICO --------------------------------------

# Preparar dados para o Stan
data_list <- list(
  N = Nx,
  T = Nt,
  f_obs = y_obs
)

# Modelo Stan usando solver numérico parametrizado em w
stan_model_code <- 
  "
data {
  int<lower=1> N;     // Número de pontos no espaço (x)
  int<lower=1> T;     // Número de passos de tempo (t)
  matrix[N, T] f_obs; // Matriz de dados observados (f(x,t))
}

parameters {
  real<lower=0, upper=0.5> w;  // Parâmetro de estabilidade numérica
  real<lower=0> sigma_e;       // Desvio padrão do ruído nos dados
}

model {
  matrix[N, T] f;  // Matriz de valores calculados de f(x,t)

  // Inicializar f com valores observados no tempo inicial
  f[, 1] = f_obs[, 1]; 

  // Diferenças finitas para calcular f(x, t+1)
  for (t in 1:(T - 1)) {
    for (x in 2:(N - 1)) {
      f[x, t+1] = (1 - 2 * w) * f[x, t] + w * (f[x+1, t] + f[x-1, t]);
    }
    
    // Condições de fronteira
    f[1, t+1] = 0;  // Fronteira esquerda
    f[N, t+1] = 0;  // Fronteira direita
  }

  // Verossimilhança: comparar f calculado com os dados observados
  for (t in 2:T) {
    f_obs[, t] ~ normal(f[, t], sigma_e);  // Modelo com ruído aditivo gaussiano
  }
  
  // Prioris
  w ~ beta(2, 6);           // Priori centrada próxima a 0.25
  sigma_e ~ cauchy(0, 0.1); // Priori para o desvio padrão do ruído
}
"

# Compilar modelo Stan
model <- stan_model(model_code = stan_model_code)


# AJUSTE DO MODELO COM HMC -------------------------------------------------------

# Definir valores iniciais para os parâmetros
init_values <- list(
  w = w_true,
  sigma_e = sigma_e
)

# Ajuste com valores iniciais específicos
fit_hmc <- sampling(model, 
                    data = data_list, 
                    init = function() init_values,
                    iter = 2000, 
                    warmup = 1000, 
                    seed = 123, 
                    chains = 1)


# SALVAR E CARREGAR RESULTADOS ---------------------------------------------------

saveRDS(fit_hmc,"Objetos/Regressao/fit_heat_numerico_hmc.rds")
fit_hmc  <- readRDS("Objetos/Regressao/fit_heat_numerico_hmc.rds")


# ANÁLISE DE RESULTADOS ----------------------------------------------------------
# Extrair amostras posteriores
fit1 <- rstan::extract(fit_hmc)

# Converter w para alpha
w_samples <- fit1$w
alpha_samples <- w_samples * (dx * dx) / dt

# Parâmetros estimados
w_pred <- median(w_samples)
alpha_pred <- median(alpha_samples)
sigma_e_pred <- median(fit1$sigma_e)

# Plotando os parâmetros w
par(mfrow=c(2,2))
plot(w_samples, type="l", xlab = "Iteration", ylab = expression(w), main="Traço do parâmetro w") 
abline(h=w_true, lty=2, lwd=2, col="red")

hist(w_samples, ylab="Densidade", xlab = expression(w), 
     prob = TRUE, main="Distribuição posterior de w", border=FALSE, col="skyblue")
abline(v=w_true, lty=2, lwd=2, col="red")

# Plotando os parâmetros alpha convertidos
plot(alpha_samples, type="l", xlab = "Iteration", ylab = expression(alpha), main="Traço do parâmetro alpha") 
abline(h=alpha, lty=2, lwd=2, col="red")

hist(alpha_samples, ylab="Densidade", xlab = expression(alpha), 
     prob = TRUE, main="Distribuição posterior de alpha", border=FALSE, col="skyblue")
abline(v=alpha, lty=2, lwd=2, col="red")

# Plotando sigma_e
par(mfrow=c(1,2))
plot(fit1$sigma_e, type="l", xlab = "Iteration", ylab = expression(sigma[e]), main="Traço do parâmetro sigma_e") 
abline(h=sigma_e, lty=2, lwd=2, col="red")

hist(fit1$sigma_e, ylab="Densidade", xlab = expression(sigma[e]), 
     prob = TRUE, main="Distribuição posterior de sigma_e", border=FALSE, col="skyblue")
abline(v=sigma_e, lty=2, lwd=2, col="red")


# ANÁLISE DE RESULTADOS ----------------------------------------------------------
# Extrair amostras posteriores

fit_hmc  <- readRDS("Objetos/Regressao/fit_heat_numerico_hmc.rds")
fit1 <- rstan::extract(fit_hmc)

# Converter w para alpha
w_samples <- fit1$w
alpha_samples <- w_samples * (dx * dx) / dt

# Parâmetros estimados
w_pred <- median(w_samples)
alpha_pred <- median(alpha_samples)
sigma_e_pred <- median(fit1$sigma_e)

# Plotando os parâmetros w
par(mfrow=c(2,2))
plot(w_samples, type="l", xlab = "Iteration", ylab = expression(w), main="Traço do parâmetro w") 
abline(h=w_true, lty=2, lwd=2, col="red")

hist(w_samples, ylab="Densidade", xlab = expression(w), 
     prob = TRUE, main="Distribuição posterior de w", border=FALSE, col="skyblue")
abline(v=w_true, lty=2, lwd=2, col="red")

# Plotando os parâmetros alpha convertidos
plot(alpha_samples, type="l", xlab = "Iteration", ylab = expression(alpha), main="Traço do parâmetro alpha") 
abline(h=alpha, lty=2, lwd=2, col="red")

hist(alpha_samples, ylab="Densidade", xlab = expression(alpha), 
     prob = TRUE, main="Distribuição posterior de alpha", border=FALSE, col="skyblue")
abline(v=alpha, lty=2, lwd=2, col="red")

# Plotando sigma_e
par(mfrow=c(1,2))
plot(fit1$sigma_e, type="l", xlab = "Iteration", ylab = expression(sigma[e]), main="Traço do parâmetro sigma_e") 
abline(h=sigma_e, lty=2, lwd=2, col="red")

hist(fit1$sigma_e, ylab="Densidade", xlab = expression(sigma[e]), 
     prob = TRUE, main="Distribuição posterior de sigma_e", border=FALSE, col="skyblue")
abline(v=sigma_e, lty=2, lwd=2, col="red")

# SUPERFÍCIE ANALÍTICA COM ALPHA VERDADEIRO -------------------------------------
# Gerar superfície usando a solução analítica com alpha verdadeiro
f_analytic_true <- matrix(0, nrow = Nx, ncol = Nt)
sigma_c <- 1  # Largura da gaussiana
for (n in 1:Nt) {
  for (i in 1:Nx) {
    f_analytic_true[i, n] <- exp(-((x[i] - L/2)^2) / (2 * sigma_c^2)) * exp(-alpha * t[n])
  }
}

# SUPERFÍCIE ANALÍTICA COM ALPHA ESTIMADO ---------------------------------------
# Gerar superfície usando a solução analítica com alpha estimado
f_analytic_est <- matrix(0, nrow = Nx, ncol = Nt)
for (n in 1:Nt) {
  for (i in 1:Nx) {
    f_analytic_est[i, n] <- exp(-((x[i] - L/2)^2) / (2 * sigma_c^2)) * exp(-alpha_pred * t[n])
  }
}

# Preparar dados para visualização
data_pred <- expand.grid(x = x, t = t)
data_pred$f_analytic_true <- as.vector(f_analytic_true)
data_pred$f_analytic_est <- as.vector(f_analytic_est)

# Visualizar superfície analítica - alpha verdadeiro
plot_ly(data_pred, x = ~x, y = ~t, z = ~f_analytic_true, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Solução Analítica (alpha verdadeiro)',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p3

# Visualizar superfície analítica - alpha estimado
plot_ly(data_pred, x = ~x, y = ~t, z = ~f_analytic_est, type = 'contour', colorscale = 'Viridis') %>%
  layout(title = 'Solução Analítica (alpha estimado)',
         xaxis = list(title = 'Posição (x)'),
         yaxis = list(title = 'Tempo (t)')) -> p4

# Comparações entre superfícies
subplot(p3, p4, nrows = 1, shareX = TRUE, shareY = TRUE) %>%
  layout(title = 'Comparação: Analítico (alpha verdadeiro) vs. Analítico (alpha estimado)')


# END OF CODE ----