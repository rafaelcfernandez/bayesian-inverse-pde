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
