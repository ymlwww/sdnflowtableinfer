function [ tau ] = characteristic_time_LRU( lambda,dI,C,tau0 )
% solve the characteristic equation for the characteristic time tau under
% LRU
% lambda: 1*F array, lambda(i) is the rateof flow i (in descending order of
% size)
% dI: average time to install a new rule after cache miss
% C: cache size
% tau0: initial guess of characteristic time
f = @(tau) characteristic_time_LRU_helper(tau,lambda,dI,C);
tau = fsolve(f,tau0);
end

function val = characteristic_time_LRU_helper(tau,lambda,dI,C)
val = sum((exp(tau*lambda)-1)./(dI*lambda+exp(tau*lambda))) - C; % LHS-RHS of the characteristic equation
end