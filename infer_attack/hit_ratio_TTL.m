function [ h ] = hit_ratio_TTL( lambda, tau, dI, policy )
% TTL approximation for the hit ratio:
h = 0;
switch policy
    case 'FIFO'
        [ h ] = hit_ratio_FIFO( lambda, tau, dI );
    case 'LRU'
        [ h ] = hit_ratio_LRU( lambda, tau, dI );
    otherwise
        error(['unknown policy: ' policy]);
end

end

