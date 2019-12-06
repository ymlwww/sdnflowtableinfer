function [ tau ] = characteristic_time_hit_ratio( h, lambda, dI, policy )
% invert the TTL approximation to compute the characteristic time (tau)
% from given probing rate (lambda), hit ratio (h) for a cache using policy
% (policy) and insertion delay (dI)
switch policy
    case 'FIFO'
        tau = h.*(1+lambda.*dI)./lambda./(1-h);        
    case 'LRU'
        tau = log((1+h.*lambda.*dI)./(1-h))./lambda;
    otherwise
        error(['unknown policy: ' policy]);        
end


end

