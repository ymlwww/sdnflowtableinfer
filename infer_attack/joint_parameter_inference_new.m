function [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference_new( hit, tau, N_flows, Fmin, Fstep, Fmax, lambdamin, lambdastep, lambdamax,...
    alphamin, alphastep, alphamax, dI, policy, trueC, trueF, trueLambda, trueAlpha )
% jointly inferring C, F, lambda, alpha by solving 4 equations with given
% hit ratio hit(1:4) and characteristic time tau(1:4)
% if trueC > 0, then error is measured against trueC; otherwise, error is
% measured against the average of the estimated C from the four
% experiments.
% if not specifying true value: trueC = 0, trueF = 0, trueLambda = 0, trueAlpha = -1. 
% even if true cache size 'trueC' is given, still cannot accurate infer F, lambda, and alpha.

if trueF > 0
    F_total = trueF;
else
    F_total = Fmin:Fstep:Fmax;
end
if trueLambda > 0
    lambda_total = trueLambda;
else
    lambda_total = lambdamin:lambdastep:lambdamax;
end
if trueAlpha >= 0
    alpha_total = trueAlpha;
else
    alpha_total = alphamin:alphastep:alphamax;
end

error = inf;
C_est = 0;
F_est = 0;
lambda_est = 0;
alpha_est = 0;
for f = 1:length(F_total)
    F = F_total(f);
    
    for l = 1:length(lambda_total)
        lambda = lambda_total(l);
        for a = 1:length(alpha_total)
            alpha = alpha_total(a);
            C = zeros(1,length(hit));
            for i=1:length(hit)
                C(i) = sum( hit_ratio_TTL( lambda*(1./(1:F).^alpha)./sum(1./(1:F).^alpha), tau(i), dI, policy ) ) + N_flows(i)*hit(i);
            end
%             error1 = sum(abs(C-round(mean(C)))); % sum absolute fitting error  % mean(C) is the estimated cache size, minimizing this minimizes the squared error in fitting mean(C) with the righthand of the characteristic equation
%             error1 = norm(C-round(mean(C)))^2; % sum squared fitting error
            if trueC > 0
                error1 = sqrt(mean((C-trueC).^2))/trueC; 
            else
                error1 = sqrt(mean((C-round(mean(C))).^2))/round(mean(C)); % normalized root mean squared error (RMSE)
            end
            if error1 < error
                error = error1;
                C_est = round(mean(C));
                F_est = F;
                lambda_est = lambda;
                alpha_est = alpha;
                
                disp(['new error ' num2str(error) ', achieved at F = ' num2str(F) ', lambda = ' num2str(lambda) ', alpha = ' num2str(alpha) ' (C = ' num2str(C_est) ')'])
            end
        end
    end
end

% if error > 10^-4
%     disp(['parameter inference failed: C = ' num2str(C_est)])
% end

end

% % Implementation 1: use Matlab's vpasolve (solve a system of nonpolynomial
% % equations) (very very slow)
% syms C  lambda alpha1
% F = 5000;
% eqns = [];
% for i=1:length(hit)
%     eqns = [eqns, sum( hit_ratio_TTL( lambda*(1./(1:F).^alpha1)./sum(1./(1:F).^alpha1), tau(i), dI, policy ) ) == C - hit(i)];
% end
% S = vpasolve(eqns,[C lambda alpha1]);