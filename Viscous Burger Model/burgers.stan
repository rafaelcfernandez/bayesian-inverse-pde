functions {
  // 1. SE Kernel: Cov(u, u)
  matrix se_kernel(vector x1, vector t1, vector x2, vector t2, real sigma2, real rx, real rt) {
    int n1 = rows(x1);
    int n2 = rows(x2);
    matrix[n1, n2] K;
    for (i in 1:n1) {
      for (j in 1:n2) {
        real dx = x1[i] - x2[j];
        real dt = t1[i] - t2[j];
        K[i,j] = sigma2 * exp(-0.5 * (square(dx)/square(rx) + square(dt)/square(rt)));
      }
    }
    return K;
  }
  
  // 2. Function-Operator: Cov(u, L_alpha u)
  matrix cov_u_Lu(vector x1, vector t1, vector x2, vector t2, real alpha, real sigma2, real rx, real rt) {
    int n1 = rows(x1);
    int n2 = rows(x2);
    matrix[n1, n2] K;
    for (i in 1:n1) {
      for (j in 1:n2) {
        real dx = x1[i] - x2[j];
        real dt = t1[i] - t2[j];
        real k_base = sigma2 * exp(-0.5 * (square(dx)/square(rx) + square(dt)/square(rt)));
        K[i,j] = k_base * (dt/square(rt) - alpha * (square(dx)/pow(rx,4) - 1/square(rx)));
      }
    }
    return K;
  }
  
  // 3. Operator-Operator: Cov(L_alpha u, L_alpha u) - CORRIGIDO
  matrix cov_Lu_Lu(vector x, vector t, real alpha, real sigma2, real rx, real rt, real jitter) {
    int n = rows(x);
    matrix[n, n] K;
    for (i in 1:n) {
      for (j in i:n) {
        real dx = x[i] - x[j];
        real dt = t[i] - t[j];
        real k_base = sigma2 * exp(-0.5 * (square(dx)/square(rx) + square(dt)/square(rt)));
        K[i,j] = k_base * (1/square(rt) - square(dt)/pow(rt,4)
                   + square(alpha) * (pow(dx,4)/pow(rx,8) - 6*square(dx)/pow(rx,6) + 3/pow(rx,4)));
        if (i != j) K[j,i] = K[i,j];
      }
      K[i,i] += jitter;
    }
    return K;
  }
}

data {
  int n_0;                    // Número de pontos boundary/initial
  int n_d;                    // Número de pontos interior observados
  vector[n_0] x_0;           // Coordenadas x dos pontos boundary
  vector[n_0] t_0;           // Coordenadas t dos pontos boundary
  vector[n_0] b_0;           // Valores conhecidos boundary/initial
  vector[n_d] x_d;           // Coordenadas x dos pontos interior observados
  vector[n_d] t_d;           // Coordenadas t dos pontos interior observados
  vector[n_d] y_d;           // Valores observados nos pontos interior
  real b_rx;                 // Prior para rx
  real b_rt;                 // Prior para rt
  real b_sigma2;             // Prior para sigma2_kernel
  real alpha_mean;           // Prior média para alpha
  real alpha_sd;             // Prior desvio para alpha
  real sigma_y_mean;         // Prior média para sigma_y
  real sigma_y_scale;        // Prior escala para sigma_y
  real jitter;               // Regularização numérica
}

parameters {
  real<lower=0> alpha;
  real<lower=0> sigma_y;
  real<lower=0> sigma2_kernel;
  real<lower=0> rx;
  real<lower=0> rt;
}

model {
  // ====================================================================
  // PASSO 1: CONSTRUIR MATRIZES COMPLETAS (TODOS OS PONTOS AMOSTRADOS)
  // ====================================================================
  
  int n_total = n_0 + n_d;
  vector[n_total] x_all;
  vector[n_total] t_all;
  
  // Concatenar coordenadas: primeiro boundary, depois interior
  x_all[1:n_0] = x_0;
  x_all[(n_0+1):n_total] = x_d;
  t_all[1:n_0] = t_0;
  t_all[(n_0+1):n_total] = t_d;
  
  // Matrizes de covariância completas
  matrix[n_total, n_total] K_uu = se_kernel(x_all, t_all, x_all, t_all, sigma2_kernel, rx, rt);
  matrix[n_total, n_total] K_uL = cov_u_Lu(x_all, t_all, x_all, t_all, alpha, sigma2_kernel, rx, rt);
  matrix[n_total, n_total] K_LL = cov_Lu_Lu(x_all, t_all, alpha, sigma2_kernel, rx, rt, jitter);
  
  // ====================================================================
  // PASSO 2: CONDICIONAMENTO B-PIGP NA PDE
  // ====================================================================
  
  matrix[n_total, n_total] Sigma_u_L = K_uu - K_uL * mdivide_left_spd(K_LL, K_uL');
  
  // ====================================================================
  // PASSO 3: PARTICIONAR EM BLOCOS BOUNDARY/INTERIOR
  // ====================================================================
  
  // Sigma_u_L já está ordenada: [boundary, interior]
  matrix[n_0, n_0] Sigma_00 = Sigma_u_L[1:n_0, 1:n_0];
  matrix[n_d, n_d] Sigma_dd = Sigma_u_L[(n_0+1):n_total, (n_0+1):n_total];
  matrix[n_d, n_0] Sigma_d0 = Sigma_u_L[(n_0+1):n_total, 1:n_0];
  
  // ====================================================================
  // PASSO 4: CONDICIONAMENTO GAUSSIANO NAS BOUNDARY CONDITIONS
  // ====================================================================
  
  // Média condicional (assumindo média zero)
  vector[n_d] mu_star = Sigma_d0 * mdivide_left_spd(Sigma_00, b_0);
  
  // Covariância condicional
  matrix[n_d, n_d] Sigma_star = Sigma_dd - Sigma_d0 * mdivide_left_spd(Sigma_00, Sigma_d0');
  
  // ====================================================================
  // PASSO 5: LIKELIHOOD DOS DADOS OBSERVADOS
  // ====================================================================
  
  // Adicionar erro de medição
  matrix[n_d, n_d] Sigma_obs = Sigma_star + diag_matrix(rep_vector(square(sigma_y), n_d));
  
  // Likelihood
  y_d ~ multi_normal_cholesky(mu_star, cholesky_decompose(Sigma_obs));
  
  // ====================================================================
  // PRIORS
  // ====================================================================
  
  alpha ~ normal(alpha_mean, alpha_sd);
  sigma_y ~ normal(sigma_y_mean, sigma_y_scale);
  sigma2_kernel ~ inv_gamma(2, b_sigma2);  
  rx ~ inv_gamma(2, b_rx);
  rt ~ inv_gamma(2, b_rt);
}
