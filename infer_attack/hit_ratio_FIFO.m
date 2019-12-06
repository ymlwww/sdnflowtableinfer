function [ h ] = hit_ratio_FIFO( lambda, tau, dI )
% TTL approximation for the hit ratio of one type of requests to a FIFO
% cache
% lambda: arrival rate of this type of requests (assumed to be Poisson)
% tau: characteristic time
% dI: average delay under a miss
h = lambda.*tau./(1+lambda.*(dI+tau));

end

