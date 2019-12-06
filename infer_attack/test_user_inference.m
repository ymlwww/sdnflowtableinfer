% Policy-aware cache size & user parameter inference (under heavy load):
% Note: this file is outdated and replaced by "test_joint_inference.m"

%% set the parameters:
C = 1000; % cache size
F = 5000; % number of distinct contents requested by background traffic (i.e., #background flows)
alpha = 0.9; % Zipf parameter of the content popularity distribution (i.e., flow size distribution)
lambda_background = 10; % background traffic rate (packets/ms)
lambda_a = 1; % maximum probing rate 
% Cinit = ones(C,1)*(-2); % initial cache state: empty cache
Cinit = [0:C-1]'; % most popular contents
dI = 20; % average time to get a new content after a miss (i.e., time to install a new rule) (ms)

p = (1./(1:F).^alpha);
%disp(lambda);
Lambda_background = lambda_background*p./sum(p); % Lambda(i): rate of the i-th largest background flow
b = 1;
time_end = 10000; % duration of each probing experiment (ms)
runs = 20; % #runs for background traffic generation
runs1 = 1; % #runs for characteristic_time_estimation

%% run probing experiments, vary b (for characteristic_time_estimation)
policy = 'FIFO';
% policy = 'LRU';
fprintf('\n\n %s:\n',policy)

Fmin = 1000; Fstep = 1000; Fmax = 10000; % parameter for grid search
lambdamin = 1; lambdastep = 1; lambdamax = 10;
alphamin = 0; alphastep = .1; alphamax = 1;
n_eqs = 4; %[2 4 6 8]; % #probing experiments
% design probing rates to have maximum diversity in hit ratios (of probes):
Lambda = zeros(1,n_eqs); % Lambda(eindx): probing rate for the eindx-th experiment
Tau_true = zeros(1,n_eqs); 
Hit_true = zeros(1,n_eqs); % computed from Tau_true
tau0=0;
[ tau_bg ] = characteristic_time( Lambda_background,dI,C,tau0,policy ); 
hit_target = .1:(.9-.1)/(n_eqs-1):.9;
for eindx = 1:n_eqs
    Lambda(eindx) = design_probing_rate(hit_target(eindx), tau_bg, dI, policy); %lambda_a*[.001 .005 .01 1]; %lambda_a*rand(1,n_eqs);
    Tau_true(eindx) = characteristic_time( [Lambda_background Lambda(eindx)],dI,C,tau0,policy ); % compute true characteristic time    
    Hit_true(eindx) = hit_ratio_TTL(Lambda(eindx), Tau_true(eindx), dI, policy);
end

B = 1; %[1 5 9 13];% 10 20 30 40]; % 50 100];

c_EST = zeros(runs,3,length(B));
f_EST = c_EST;
lm_EST = c_EST;
a_EST = c_EST;
t_EST = zeros(runs,3,length(B),n_eqs);
h_EST = t_EST;
e_thresh = 10^(-4); 
for nindx = 1:length(B)
    b = B(nindx); % to estimate F, alpha, lambda_background, C, we need 4 equations at least    
    for rindx = 1:runs % #successful probing rounds
    rindx1 = 1;
    while 1
        disp(' ')
        disp(['b = ' num2str(b) ', run ' num2str(rindx) ': trial ' num2str(rindx1) '...'])
        tic
        trace = exp_trace(F, alpha, lambda_background, time_end); % background traffic
        Tau = zeros(3,1,n_eqs);
        Hit = Tau;
        abort = 0;
        for eindx = 1:n_eqs
            %         [ tau ] = characteristic_time( [Lambda_background Lambda(eindx)],dI,C,tau0,policy ); % compute true characteristic time
            tau = Tau_true(eindx);
            probes = exp_trace(1, alpha, Lambda(eindx), time_end);
            probes(:,2) = -1;
            mergetrace = cat(1,trace,probes);
            mergetrace = sortrows(mergetrace,1);
            %             % Method 1: measure hit ratio, calculate characteristic time
            %             cache.C = Cinit;
            %             cache.pend = zeros(0,2);
            %             hits = 0;
            %             for i=1:length(mergetrace(:,1))
            %                 [cache, ishit] = CacheAdd_withdelay( cache, mergetrace(i,1), mergetrace(i,2), dI, policy);
            %                 if (mergetrace(i,2) == -1)
            %                     hits = hits + ishit;
            %                 end
            %             end
            %             Hit(1,eindx) = hits/size(probes,1);
            %             Tau(1,eindx) = characteristic_time_hit_ratio( Hit(1,eindx), Lambda(eindx), dI, policy );
            %             disp(['at rate ' num2str(Lambda(eindx)) ':'])
            %             disp(['method 1: hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ' (true: ' num2str(tau) ')'])
            % Method 2: estimate characteristic time, calculate hit ratio
            [ T ] = characteristic_time_estimation( mergetrace, tau+10, b, tau*.01, Cinit, dI, policy );
            if eindx > 1 && (T >= Tau(2,1,eindx-1) || T < Tau(2,1,eindx-1)-10*(Tau_true(eindx-1)-Tau_true(eindx))) % characteristic time should be decreasing as probing rate increases
                abort = 1;
                break;
            end
            Tau(2,1,eindx) = T;
            Hit(2,1,eindx) = hit_ratio_TTL(Lambda(eindx), T, dI, policy); % calculated hit ratio
            %             Hit(2,eindx) = hits/size(probes,1); % measured hit ratio
            %             disp(['method 2: hit probability = ' num2str(Hit_EST(2,rindx,nindx,eindx)) ', tau = ' num2str(T) ' (true: ' num2str(tau) ')'])
            % benchmark: both are calculated
%             Tau(3,1,eindx) = tau;
%             Hit(3,1,eindx) = hit_ratio_TTL(Lambda(eindx), tau, dI, policy);
        end
        if abort 
            disp('aborted'); disp(' ')
        else% if the probing experiments complete successfully            
        % print the result:
        disp(' ')
        for eindx = 1:n_eqs
            disp(['at rate ' num2str(Lambda(eindx)) ':'])
            %             disp(['method 1:   hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ])
            disp(['method 2:   hit probability = ' num2str(Hit(2,1,eindx)) ', tau = ' num2str(Tau(2,1,eindx)) ])
            disp(['true value: hit probability = ' num2str(Hit(3,1,eindx)) ', tau = ' num2str(Tau(3,1,eindx)) ])
        end
%         % solve equations:
%         disp(' ')
%         disp('Method 2: estimate characteristic time, calculate hit ratio')
%         [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference( Hit(2,1,:), Tau(2,1,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
%         if error < e_thresh % successfully approximate the equations
%             c_EST(rindx,2,nindx) = C_est;
%             f_EST(rindx,2,nindx) = F_est;
%             lm_EST(rindx,2,nindx) = lambda_est;
%             a_EST(rindx,2,nindx) = alpha_est;
            t_EST(rindx,2,nindx,:) = reshape(Tau(2,1,:),1,1,1,n_eqs);
            h_EST(rindx,2,nindx,:) = reshape(Hit(2,1,:),1,1,1,n_eqs);
            break;
%         end
        end
        rindx1 = rindx1 + 1;
        disp('takes ' num2str(toc) ' seconds')
    end%while 1
    end% for rindx = 1:runs
%     disp(' ')
%     disp('Ideal: both hit ratio and characteristic time are calculated')
%     [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference( Hit(3,1,:), Tau(3,1,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
%     % FIFO:
%     % n_eqs = 2: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
%     % n_eqs = 4: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
%     % n_eqs = 6: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
%     c_EST(:,3,nindx) = C_est;
%     f_EST(:,3,nindx) = F_est;
%     lm_EST(:,3,nindx) = lambda_est;
%     a_EST(:,3,nindx) = alpha_est;
%     t_EST(3,nindx,:) = Tau(3,1,:);
%     h_EST(3,nindx,:) = Hit(3,1,:);
end% for nindx = 1:length(B)
save(['data/user_inference_runs_' policy '.mat'], 'B', 'h_EST', 't_EST'); %, 'c_est','f_est','lm_est','a_est','C','F','lambda_background','alpha');        
% save(['data/user_inference_b_' policy '.mat'], 'B', 'h_EST', 't_EST', 'c_EST','f_EST','lm_EST','a_EST','C','F','lambda_background','alpha');
%% if solving equations in the loop of probing experiments:
if 0
figure;
c_est = reshape(mean(c_EST,1),3,length(B));
subplot(2,2,1)
plot(B, c_est(2,:), 'b*-',...
    B, c_est(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated C')
legend('estimated','true')

subplot(2,2,2)
f_est = reshape(mean(f_EST,1),3,length(B));
plot(B, f_est(2,:), 'b*-',...
    B, f_est(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated F')
legend('estimated','true')

subplot(2,2,3)
lm_est = reshape(mean(lm_EST,1),3,length(B));
plot(B, lm_est(2,:), 'b*-',...
    B, lm_est(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated \lambda')
legend('estimated','true')

subplot(2,2,4)
a_est = reshape(mean(a_EST,1),3,length(B));
plot(B, a_est(2,:), 'b*-',...
    B, a_est(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated \alpha')
legend('estimated','true')    
end
%% after probing experiment, try gradually increasing #Monte Carlo runs to get better estimates and then solve the equations:
% load(['data/user_inference_runs_' policy '.mat']);
nindx = 1; % index in B
c_est = zeros(3,runs); %c_est(i,r): mean(c_EST(1:r,i,nindx))
f_est = c_est;
lm_est = c_est;
a_est = c_est;
for r=1:runs % average over r runs
    tau_est = mean(t_EST(1:r,2,nindx,:),1);
    hit_est = mean(h_EST(1:r,2,nindx,:),1);
    [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference( hit_est, tau_est, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
    c_est(2,r) = C_est;
    f_est(2,r) = F_est;
    lm_est(2,r) = lambda_est;
    a_est(2,r) = alpha_est;    
    disp(' ')
%     figure;
%     subplot(1,2,1)
%     plot(Lambda, reshape(tau_est,size(Lambda)), 'b*',...
%         Lambda, Tau_true, 'k--', 'LineWidth', 1.5);
%     xlabel('probing rate')
%     ylabel('characteristic time')
%     legend('estimated','true')
%     subplot(1,2,2)
%     plot(Lambda, reshape(hit_est,size(Lambda)), 'b*',...
%         Lambda, Hit_true, 'k--', 'LineWidth', 1.5); 
%     xlabel('probing rate')
%     ylabel('hit ratio')
%     legend('estimated','true')
end

figure;
    subplot(2,2,1)
    plot(1:runs, c_est(2,:), 'b*-',...
        1:runs, C*ones(1,runs), 'k--', 'LineWidth', 1.5);
    xlabel('#Monte Carlo runs')
    ylabel('estimated C')
    legend('estimated','true')
    
    subplot(2,2,2)
    plot(1:runs, f_est(2,:), 'b*-',...
        1:runs, F*ones(1,runs), 'k--', 'LineWidth', 1.5);
    xlabel('#Monte Carlo runs')
    ylabel('estimated F')
    legend('estimated','true')
    
    subplot(2,2,3)
    plot(1:runs, lm_est(2,:), 'b*-',...
        1:runs, lambda_background*ones(1,runs), 'k--', 'LineWidth', 1.5);
    xlabel('#Monte Carlo runs')
    ylabel('estimated \lambda')
    legend('estimated','true')
    
    subplot(2,2,4)
    plot(1:runs, a_est(2,:), 'b*-',...
        1:runs, alpha*ones(1,runs), 'k--', 'LineWidth', 1.5);
    xlabel('#Monte Carlo runs')
    ylabel('estimated \alpha')
    legend('estimated','true')
title(['b = ' num2str(B(nindx))])


%% start the Monte Carlo run:    
C_EST = zeros(3,runs,length(B));
F_EST = C_EST;
Lambda_EST = C_EST;
Alpha_EST = C_EST;
Tau_EST = zeros(3,runs,length(B),n_eqs);
Hit_EST = zeros(3,runs,length(B),n_eqs);
for rindx = 1:runs
    disp(['run ' num2str(rindx) '...'])
    trace = exp_trace(F, alpha, lambda_background, time_end); % background traffic
    for eindx = 1:n_eqs
        [ tau ] = characteristic_time( [Lambda_background Lambda(eindx)],dI,C,tau0,policy ); % compute true characteristic time
        probes = exp_trace(1, alpha, Lambda(eindx), time_end);
        probes(:,2) = -1;
        mergetrace = cat(1,trace,probes);
        mergetrace = sortrows(mergetrace,1);    
        for nindx = 1:length(B)
            b = B(nindx); % to estimate F, alpha, lambda_background, C, we need 4 equations at least
            disp(['b = ' num2str(b) ': '])
            %         Tau = zeros(2,n_eqs); % 1: method 1, 2: method 2
            %         Hit = zeros(2,n_eqs);
            %         Tau_true = zeros(1,n_eqs); % true characteristic time under each probing rate
            %         Hit_true = zeros(1,n_eqs); % true hit probability by the TTL approximation
            
            %             % Method 1: measure hit ratio, calculate characteristic time
            cache.C = Cinit;
            cache.pend = zeros(0,2);
            hits = 0;
            for i=1:length(mergetrace(:,1))
                [cache, ishit] = CacheAdd_withdelay( cache, mergetrace(i,1), mergetrace(i,2), dI, policy);
                if (mergetrace(i,2) == -1)
                    hits = hits + ishit;
                end
            end
            %             Hit(1,eindx) = hits/size(probes,1);
            %             Tau(1,eindx) = characteristic_time_hit_ratio( Hit(1,eindx), Lambda(eindx), dI, policy );
            %             disp(['at rate ' num2str(Lambda(eindx)) ':'])
            %             disp(['method 1: hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ' (true: ' num2str(tau) ')'])
            % Method 2: estimate characteristic time, calculate hit ratio
            [ T ] = characteristic_time_estimation( mergetrace, tau+10, b, tau*.01, Cinit, dI, policy );
            Tau_EST(2,rindx,nindx,eindx) = T;
            Hit_EST(2,rindx,nindx,eindx) = hit_ratio_TTL(Lambda(eindx), T, dI, policy); % calculated hit ratio
            %             Hit(2,eindx) = hits/size(probes,1); % measured hit ratio
            disp(['method 2: hit probability = ' num2str(Hit_EST(2,rindx,nindx,eindx)) ', tau = ' num2str(T) ' (true: ' num2str(tau) ')'])
            % benchmark: both are calculated
            Tau_EST(3,rindx,nindx,eindx) = tau;
            Hit_EST(3,rindx,nindx,eindx) = hit_ratio_TTL(Lambda(eindx), tau, dI, policy);
        end
    end
        %% print the result:
        disp(' ')
        for nindx = 1:length(B)
        disp(['b = ' num2str(B(nindx)) ':'])
        for eindx = 1:n_eqs
            disp(['at rate ' num2str(Lambda(eindx)) ':'])
%             disp(['method 1:   hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ])
            disp(['method 2:   hit probability = ' num2str(Hit_EST(2,rindx,nindx,eindx)) ', tau = ' num2str(Tau_EST(2,rindx,nindx,eindx)) ])
            disp(['true value: hit probability = ' num2str(Hit_EST(3,rindx,nindx,eindx)) ', tau = ' num2str(Tau_EST(3,rindx,nindx,eindx)) ])
        end
        disp(' ')
        end        
        %  FIFO: (original tau = 265.4647 ms)
        % at rate 0.001: hit probability = 0.15909, tau = 192.973 (true: 265.1057)
        % at rate 0.005: hit probability = 0.53719, tau = 255.3571 (true: 265.1057)
        % at rate 0.01: hit probability = 0.69781, tau = 277.1053 (true: 265.1057)
        % at rate 1: hit probability = 0.92579, tau = 261.9827 (true: 265.1057)
        % make Tau's ordered
%         Tau(1,:) = sort(Tau(1,:),'descend');
%         Tau(2,:) = sort(Tau(2,:),'descend');

end% for rindx        
save(['data/user_inference_b_' policy '.mat'], 'B', 'Hit_EST', 'Tau_EST', 'C_EST','F_EST','Lambda_EST','Alpha_EST','C','F','lambda_background','alpha');

%% solve equations:
Fmin = 1000; Fstep = 1000; Fmax = 10000;
lambdamin = 1; lambdastep = 1; lambdamax = 10;
alphamin = 0; alphastep = .1; alphamax = 1;
%% average the Tau and Hit, then solve equations:
% Can tau and hit ratio be estimated accurately? Yes, almost accurate
t_EST = reshape(mean(Tau_EST,2),3,length(B),n_eqs);
h_EST = reshape(mean(Hit_EST,2),3,length(B),n_eqs);

nindx=4; 
figure;
plot(Lambda, reshape(t_EST(2,nindx,:),size(Lambda)), '*-',...
    Lambda, reshape(t_EST(3,nindx,:),size(Lambda)), 'k--', 'LineWidth', 1.5);
xlabel('probing rate')
ylabel('characteristic time')
legend('estimated','true')
title(['b = ' num2str(B(nindx))])
figure;
plot(Lambda, reshape(h_EST(2,nindx,:),size(Lambda)), '*-',...
    Lambda, reshape(h_EST(3,nindx,:),size(Lambda)), 'k--', 'LineWidth', 1.5);
xlabel('probing rate')
ylabel('hit ratio')
legend('estimated','true')
title(['b = ' num2str(B(nindx))])


%% but can C, F, lambda_background, alpha be estimated accurately from the estimated tau and hit ratio? No! (the equations are numerically unstable)
c_EST = zeros(3,length(B));
f_EST = c_EST;
lm_EST = c_EST;
a_EST = c_EST;
for nindx = 1:length(B)
    b = B(nindx); % to estimate F, alpha, lambda_background, C, we need 4 equations at least
    disp(['b = ' num2str(b) ': '])
    %%
    %         disp(' ')
    %         disp('Method 1: measure hit ratio, calculate characteristic time')
    %         [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit(1,:), Tau(1,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
    %         % FIFO:
    %         % n_eqs = 2: F = 1000, lambda = 1, alpha = 1 (C = 106)
    %         % n_eqs = 4: F = 1000, lambda = 1, alpha = 1 (C = 121)
    %         % n_eqs = 6: F = 1000, lambda = 1, alpha = 1 (C = 106)
    %         C_EST(1,rindx,nindx) = C_est;
    %         F_EST(1,rindx,nindx) = F_est;
    %         Lambda_EST(1,rindx,nindx) = lambda_est;
    %         Alpha_EST(1,rindx,nindx) = alpha_est;
    %%
    disp(' ')
    disp('Method 2: estimate characteristic time, calculate hit ratio')
    [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference( h_EST(2,nindx,:), t_EST(2,nindx,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
    e_thresh = 10^(-5);
    % FIFO:
    % n_eqs = 2: F = 3200, lambda = 10, alpha = 0.6 (C = 1249)
    % n_eqs = 4: F = 3900, lambda = 6, alpha = 0.5 (C = 1041)
    % n_eqs = 6: F = 1900, lambda = 1, alpha = 1 (C = 137)
    c_EST(2,nindx) = C_est;
    f_EST(2,nindx) = F_est;
    lm_EST(2,nindx) = lambda_est;
    a_EST(2,nindx) = alpha_est;
    %%
    disp(' ')
    disp('Ideal: both hit ratio and characteristic time are calculated')
    [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( h_EST(3,nindx,:), t_EST(3,nindx,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
    % FIFO:
    % n_eqs = 2: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
    % n_eqs = 4: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
    % n_eqs = 6: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
    c_EST(3,nindx) = C_est;
    f_EST(3,nindx) = F_est;
    lm_EST(3,nindx) = lambda_est;
    a_EST(3,nindx) = alpha_est;
    
end% for nindx
%%
figure;
subplot(2,2,1)
plot(B, c_EST(2,:), 'b*-',...
    B, c_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated C')
legend('estimated','true')

subplot(2,2,2)
plot(B, f_EST(2,:), 'b*-',...
    B, f_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated F')
legend('estimated','true')

subplot(2,2,3)
plot(B, lm_EST(2,:), 'b*-',...
    B, lm_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated \lambda')
legend('estimated','true')

subplot(2,2,4)
plot(B, a_EST(2,:), 'b*-',...
    B, a_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated \alpha')
legend('estimated','true')

%% solve each Monte Carlo run independently:
for rindx = 1:runs
    disp(['run ' num2str(rindx) '...'])
    for nindx = 1:length(B)    
        b = B(nindx); % to estimate F, alpha, lambda_background, C, we need 4 equations at least
        disp(['b = ' num2str(b) ': '])
        %%
%         disp(' ')
%         disp('Method 1: measure hit ratio, calculate characteristic time')
%         [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit(1,:), Tau(1,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
%         % FIFO: 
%         % n_eqs = 2: F = 1000, lambda = 1, alpha = 1 (C = 106)
%         % n_eqs = 4: F = 1000, lambda = 1, alpha = 1 (C = 121)
%         % n_eqs = 6: F = 1000, lambda = 1, alpha = 1 (C = 106)
%         C_EST(1,rindx,nindx) = C_est;
%         F_EST(1,rindx,nindx) = F_est;
%         Lambda_EST(1,rindx,nindx) = lambda_est;
%         Alpha_EST(1,rindx,nindx) = alpha_est;
        %%
        disp(' ')
        disp('Method 2: estimate characteristic time, calculate hit ratio')
        [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit_EST(2,rindx,nindx,:), Tau_EST(2,rindx,nindx,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
        % FIFO: 
        % n_eqs = 2: F = 3200, lambda = 10, alpha = 0.6 (C = 1249)
        % n_eqs = 4: F = 3900, lambda = 6, alpha = 0.5 (C = 1041)
        % n_eqs = 6: F = 1900, lambda = 1, alpha = 1 (C = 137)
        C_EST(2,rindx,nindx) = C_est;
        F_EST(2,rindx,nindx) = F_est;
        Lambda_EST(2,rindx,nindx) = lambda_est;
        Alpha_EST(2,rindx,nindx) = alpha_est;
        %%
        disp(' ')
        disp('Ideal: both hit ratio and characteristic time are calculated')
        [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit_EST(3,rindx,nindx,:), Tau_EST(3,rindx,nindx,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
        % FIFO: 
        % n_eqs = 2: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
        % n_eqs = 4: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
        % n_eqs = 6: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
        C_EST(3,rindx,nindx) = C_est;
        F_EST(3,rindx,nindx) = F_est;
        Lambda_EST(3,rindx,nindx) = lambda_est;
        Alpha_EST(3,rindx,nindx) = alpha_est;
        
    end% for nindx
end% for rindx

save(['data/user_inference_b_' policy '.mat'], 'B', 'Hit_EST', 'Tau_EST', 'C_EST','F_EST','Lambda_EST','Alpha_EST','C','F','lambda_background','alpha');

%% plot results:
load(['data/user_inference_b_' policy '.mat']);
C_EST = reshape(mean(C_EST,2),3,length(B)); %zeros(3,runs,length(B));
F_EST = reshape(mean(F_EST,2),3,length(B));
Lambda_EST = reshape(mean(Lambda_EST,2),3,length(B));
Alpha_EST = reshape(mean(Alpha_EST,2),3,length(B));

figure;
subplot(2,2,1)
plot(B, C_EST(2,:), 'b*-',...
    B, C_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated C')
legend('estimated','true')

subplot(2,2,2)
plot(B, F_EST(2,:), 'b*-',...
    B, F_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated F')
legend('estimated','true')

subplot(2,2,3)
plot(B, Lambda_EST(2,:), 'b*-',...
    B, Lambda_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated \lambda')
legend('estimated','true')

subplot(2,2,4)
plot(B, Alpha_EST(2,:), 'b*-',...
    B, Alpha_EST(3,:), 'k--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('estimated \alpha')
legend('estimated','true')

%% run probing experiments, vary n_eqs (#experiments at different probing rates)
policy = 'FIFO';
% policy = 'LRU';
fprintf('\n\n %s:\n',policy)
N_eqs = 4; %[2 4 6 8]; % #probing experiments
% design probing rates to have maximum diversity in hit ratios (of probes):
Lambda_designed = cell(1,length(N_eqs)); % Lambda{nindx}: probing rates when running N_eqs(nindx) experiments
tau0=0;
[ tau_bg ] = characteristic_time( Lambda_background,dI,C,tau0,policy ); 
for nindx = 1:length(N_eqs)    
    n_eqs = N_eqs(nindx);    
    hit_target = .1:(.9-.1)/(n_eqs-1):.9;
    Lambda_designed{nindx} = zeros(1,n_eqs);
    for eindx = 1:n_eqs
        Lambda_designed{nindx}(eindx) = design_probing_rate(hit_target(eindx), tau_bg, dI, policy); %lambda_a*[.001 .005 .01 1]; %lambda_a*rand(1,n_eqs);
    end
end
% start the Monte Carlo run:    
C_EST = zeros(3,runs,N_eqs);
F_EST = C_EST;
Lambda_EST = C_EST;
Alpha_EST = C_EST;
for rindx = 1:runs
    trace = exp_trace(F, alpha, lambda_background, time_end); % background traffic
    for nindx = 1:length(N_eqs)    
        n_eqs = N_eqs(nindx); % to estimate F, alpha, lambda_background, C, we need 4 equations at least
        Lambda = Lambda_designed{nindx};
        Tau = zeros(2,n_eqs); % 1: method 1, 2: method 2
        Hit = zeros(2,n_eqs);
        Tau_true = zeros(1,n_eqs); % true characteristic time under each probing rate
        Hit_true = zeros(1,n_eqs); % true hit probability by the TTL approximation
        
        for eindx = 1:n_eqs
            probes = exp_trace(1, alpha, Lambda(eindx), time_end);
            probes(:,2) = -1;
            mergetrace = cat(1,trace,probes);
            mergetrace = sortrows(mergetrace,1);
            % Method 1: measure hit ratio, calculate characteristic time
            tic
            cache.C = Cinit;
            cache.pend = zeros(0,2);
            hits = 0;
            for i=1:length(mergetrace(:,1))
                [cache, ishit] = CacheAdd_withdelay( cache, mergetrace(i,1), mergetrace(i,2), dI, policy);
                if (mergetrace(i,2) == -1)
                    hits = hits + ishit;
                end
            end
            Hit(1,eindx) = hits/size(probes,1);
            Tau(1,eindx) = characteristic_time_hit_ratio( Hit(1,eindx), Lambda(eindx), dI, policy );            
            [ tau ] = characteristic_time( [Lambda_background Lambda(eindx)],dI,C,tau0,policy ); % compute true characteristic time
            disp(['at rate ' num2str(Lambda(eindx)) ':'])
            disp(['method 1: hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ' (true: ' num2str(tau) ')'])
            % Method 2: estimate characteristic time, calculate hit ratio
            [ T ] = characteristic_time_estimation( mergetrace, tau+10, b, tau*.01, Cinit, dI, policy );
            Tau(2,eindx) = T;
            Hit(2,eindx) = hit_ratio_TTL(Lambda(eindx), T, dI, policy);
            disp(['method 2: hit probability = ' num2str(Hit(2,eindx)) ', tau = ' num2str(Tau(2,eindx)) ' (true: ' num2str(tau) ')'])
            % benchmark: both are calculated
            Tau_true(eindx) = tau;
            Hit_true(eindx) = hit_ratio_TTL(Lambda(eindx), tau, dI, policy);
        end
        %% print the result:
        disp(' ')
        for eindx = 1:n_eqs
            disp(['at rate ' num2str(Lambda(eindx)) ':'])
            disp(['method 1:   hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ])
            disp(['method 2:   hit probability = ' num2str(Hit(2,eindx)) ', tau = ' num2str(Tau(2,eindx)) ])
            disp(['true value: hit probability = ' num2str(Hit_true(eindx)) ', tau = ' num2str(Tau_true(eindx)) ])
        end
        disp(' ')
        %  FIFO: (original tau = 265.4647 ms)
        % at rate 0.001: hit probability = 0.15909, tau = 192.973 (true: 265.1057)
        % at rate 0.005: hit probability = 0.53719, tau = 255.3571 (true: 265.1057)
        % at rate 0.01: hit probability = 0.69781, tau = 277.1053 (true: 265.1057)
        % at rate 1: hit probability = 0.92579, tau = 261.9827 (true: 265.1057)
        % make Tau's ordered
%         Tau(1,:) = sort(Tau(1,:),'descend');
%         Tau(2,:) = sort(Tau(2,:),'descend');
        
        %% solve equations:
        Fmin = 1000; Fstep = 100; Fmax = 5000;
        lambdamin = 1; lambdastep = 1; lambdamax = 10;
        alphamin = 0; alphastep = .1; alphamax = 1;
        %%
        disp(' ')
        disp('Method 1: measure hit ratio, calculate characteristic time')
        [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit(1,:), Tau(1,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
        % FIFO: 
        % n_eqs = 2: F = 1000, lambda = 1, alpha = 1 (C = 106)
        % n_eqs = 4: F = 1000, lambda = 1, alpha = 1 (C = 121)
        % n_eqs = 6: F = 1000, lambda = 1, alpha = 1 (C = 106)
        C_EST(1,rindx,nindx) = C_est;
        F_EST(1,rindx,nindx) = F_est;
        Lambda_EST(1,rindx,nindx) = lambda_est;
        Alpha_EST(1,rindx,nindx) = alpha_est;
        %%
        disp(' ')
        disp('Method 2: estimate characteristic time, calculate hit ratio')
        [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit(2,:), Tau(2,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
        % FIFO: 
        % n_eqs = 2: F = 3200, lambda = 10, alpha = 0.6 (C = 1249)
        % n_eqs = 4: F = 3900, lambda = 6, alpha = 0.5 (C = 1041)
        % n_eqs = 6: F = 1900, lambda = 1, alpha = 1 (C = 137)
        C_EST(2,rindx,nindx) = C_est;
        F_EST(2,rindx,nindx) = F_est;
        Lambda_EST(2,rindx,nindx) = lambda_est;
        Alpha_EST(2,rindx,nindx) = alpha_est;
        %%
        disp(' ')
        disp('Ideal: both hit ratio and characteristic time are calculated')
        [ C_est, F_est, lambda_est, alpha_est ] = joint_parameter_inference( Hit_true, Tau_true, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
        % FIFO: 
        % n_eqs = 2: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
        % n_eqs = 4: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
        % n_eqs = 6: F = 5000, lambda = 10, alpha = 0.9 (C = 1000)
        C_EST(3,rindx,nindx) = C_est;
        F_EST(3,rindx,nindx) = F_est;
        Lambda_EST(3,rindx,nindx) = lambda_est;
        Alpha_EST(3,rindx,nindx) = alpha_est;
        
    end% for nindx
end% for rindx

save(['data/user_inference_Neqs_' policy '.mat'], 'N_eqs', 'C_EST','F_EST','Lambda_EST','Alpha_EST','C','F','lambda_background','alpha');
