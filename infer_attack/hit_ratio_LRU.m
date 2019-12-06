function [ h ] = hit_ratio_LRU( lambda, tau, dI )
% TTL approximation for the hit ratio of one type of requests to a LRU
% cache
% lambda: arrival rate of this type of requests (assumed to be Poisson)
% tau: characteristic time
% dI: average delay under a miss
h = (exp(lambda.*tau)-1) ./ (lambda.*dI + exp(lambda.*tau));

end

