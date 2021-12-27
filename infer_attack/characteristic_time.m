function [ tau ] = characteristic_time( lambda, dI,C,tau0,policy )
% lambda: 1*F array, lambda(i) is the rate of flow i 
% dI: average time to install a new rule after cache miss
% C: cache size
% tau0: initial guess of characteristic time
switch policy
    case 'FIFO'
        [ tau ] = characteristic_time_FIFO( lambda, dI,C,tau0);
    case 'LRU'
        [ tau ] = characteristic_time_LRU( lambda, dI,C,tau0);
    otherwise
end

end

