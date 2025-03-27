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
