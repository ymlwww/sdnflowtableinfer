function [ tau ] = characteristic_time_FIFO( lambda,dI,C,tau0 )
% solve the characteristic equation for the characteristic time tau under
% FIFO
% lambda: 1*F array, lambda(i) is the rate of flow i 
% dI: average time to install a new rule after cache miss
% C: cache size
% tau0: initial guess of characteristic time
f = @(tau) characteristic_time_FIFO_helper(tau,lambda,dI,C);
tau = fsolve(f,tau0);
end

function val = characteristic_time_FIFO_helper(tau,lambda,dI,C)
val = sum(tau*lambda./(1+(dI+tau)*lambda)) - C; % LHS-RHS of the characteristic equation
end