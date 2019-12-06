function [h_fifo, h_lru] = hit_ratio_with_attack_theory(cachesize, lm_background, alpha, F, lm_attack, dI )
% predict the hit ratio for background traffic for a FIFO or LRU
% cache, under an attack of length(lm_attack) flows of rates in lm_attack
% (1*n array for n attack flows), using the TTL approximation
% alpha=0.9;
% F = 5000;
% dI = 0; % new rule installation time (~20 ms)
tau0 = 0; % initial guess of characteristic time
p = (1./(1:F).^alpha);
p = p./sum(p); % Zipf popularity distribution
lambda = [lm_attack lm_background*p]; % rates of all the flows
% FIFO:
[ tau ] = characteristic_time( lambda, dI,cachesize,tau0,'FIFO' );
h = hit_ratio_FIFO(lm_background*p, tau, dI );
h_fifo = sum(h.*p); % avg hit ratio
% LRU:
[ tau ] = characteristic_time( lambda, dI,cachesize,tau0,'LRU' );
h = hit_ratio_LRU(lm_background*p, tau, dI );
h_lru = sum(h.*p); % avg hit ratio

end