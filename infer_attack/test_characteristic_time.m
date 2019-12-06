% TTL approximation-based policy inference in heavily loaded network:

%% set the parameters and generate trace:
C = 1000; % cache size
F = 5000; % number of distinct contents requested by background traffic (i.e., #background flows)
alpha = 0.9; % Zipf parameter of the content popularity distribution (i.e., flow size distribution)
lambda_background = 10; % background traffic rate (packets/ms)
lambda_a = .001; % minimum probing rate (to be optimized later by lambda_a1)
b = 100; % #repetitions in Characteristic Time Estimation
delta = 0.5; % tolerance interval in Characteristic Time Estimation (ms) (not used)
% Cinit = ones(C,1)*(-2); % initial cache state: empty cache
Cinit = [0:C-1]'; % most popular contents
dI = 20; % average time to get a new content after a miss (i.e., time to install a new rule) (ms)

p = (1./(1:F).^alpha);
%disp(lambda);
Lambda = lambda_background*p./sum(p); % Lambda(i): rate of the i-th largest background flow

time_end = 10000; % duration of generated trace (ms)
runs = 10;

%% estimate the characteristic time:
% policy = 'FIFO';
policy = 'LRU';
fprintf('\n\n %s:\n',policy)
tau0=0;
% true value:
[ tau ] = characteristic_time( Lambda,dI,C,tau0,policy ); % compute true characteristic time

B = [1 10 50 100];
Tau_est = zeros(length(B),runs);
for r=1:runs
    disp(' '); disp(['run ' num2str(r) '...'])
    trace = exp_trace(F, alpha, lambda_background, time_end); % background traffic
    for bindx = 1:length(B)        
        b = B(bindx);
        % estimating TTL:
        [ T ] = characteristic_time_estimation( trace, tau+1, b, tau*.01, Cinit, dI, policy ); % tolerance: 1% of true characteristic time
        disp(['b = ' num2str(b) ': calculated = ' num2str(tau) ', estimated = ' num2str(T)])
        Tau_est(bindx,r) = T;
    end
end
% save(['data/tau_b_' policy '.mat'], 'Tau_est', 'B', 'tau');
%%
figure; % accuracy of characteristic time estimation
load(['data/tau_b_FIFO.mat']);
tau_fifo = tau;
tau_est_fifo = reshape(mean(Tau_est,2), size(B));
load(['data/tau_b_LRU.mat']);
tau_lru = tau;
tau_est_lru = reshape(mean(Tau_est,2), size(B));
plot(B, tau_est_fifo, 'rs-',...    
    B, tau_fifo*ones(size(B)), 'm--',...
    B, tau_est_lru, 'bo-',...
    B, tau_lru*ones(size(B)), 'c--', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('characteristic time (ms)')
legend('FIFO: estimated', '          true', 'LRU: estimated', '          true')

%% choose lambda_a to maximize the gap between predicted h_FIFO and h_LRU:
% policy = 'FIFO';
policy = 'LRU'; 
load(['data/tau_b_' policy '.mat']); % tau: true characteristic time
tau_est = reshape(mean(Tau_est,2), size(B));
probing_rate = zeros(size(B));
h_ideal = zeros(2,length(B)); % h_ideal(1,:): hit_ratio_FIFO based on tau; h_ideal(2,:): hit_ratio_LRU based on tau
h_predicted = h_ideal; % based on T
h_estimated = zeros(1,length(B)); % actual hit ratio

time_end = 100000;
trace = exp_trace(F, alpha, lambda_background, time_end); % background traffic
for bindx = 1:length(B)
    disp(['b = ' num2str(B(bindx)) ':'])
    T = tau_est(bindx); % T: estimated characteristic time    
    % % manually check if the selected probing rate maximizes the gap:
    % Lambda_a = .001:.01:2;
    % f = abs(hit_ratio_FIFO(Lambda_a,T,dI)-hit_ratio_LRU(Lambda_a,T,dI));
    % figure;
    % plot(Lambda_a,f);    
    f = @(lambda1) (-abs(hit_ratio_FIFO(lambda1,T,dI)-hit_ratio_LRU(lambda1,T,dI)));
    % lambda_a1 = fminsearch(f,lambda_a);
    lambda_a1 = fminbnd(f,0,lambda_background/10);
    disp(['probing at rate ' num2str(lambda_a1) '...'])
    lambda_a1 = max(lambda_a1,lambda_a);
    probing_rate(bindx) = lambda_a1; 
    %% check values for ideal prediction (using true characteristic time) and actual prediction of hit probabilities:
    % fprintf('\n hit probability: \n')
    % disp(['           FIFO        LRU'])
    % disp(['true:      ' num2str(hit_ratio_FIFO(lambda_a1,tau,dI)) '     ' num2str(hit_ratio_LRU(lambda_a1,tau,dI)) ])
    % disp(['predicted: ' num2str(hit_ratio_FIFO(lambda_a1,T,dI)) '     ' num2str(hit_ratio_LRU(lambda_a1,T,dI)) ])    
    h_ideal(1,bindx) = hit_ratio_FIFO(lambda_a1,tau,dI); 
    h_ideal(2,bindx) = hit_ratio_LRU(lambda_a1,tau,dI);
    h_predicted(1,bindx) = hit_ratio_FIFO(lambda_a1,T,dI);
    h_predicted(2,bindx) = hit_ratio_LRU(lambda_a1,T,dI);    
    
    %% measure the actual hit probability:
    probes = exp_trace(1, alpha, lambda_a1, time_end); % probing traffic as a Poisson process
    probes(:,2) = -1; % probes are for a new content
    n_probes = length(probes(:,1));
    mergetrace = cat(1,trace,probes);
    mergetrace = sortrows(mergetrace,1);
    cache.C = [0:C-1]'; % ones(C,1)*(-2); % empty cache
    cache.pend = zeros(0,2);
    hits = 0;
    for i=1:length(mergetrace(:,1))
        [cache, ishit] = CacheAdd_withdelay( cache, mergetrace(i,1), mergetrace(i,2), dI, policy);
        if (mergetrace(i,2) == -1)
            hits = hits + ishit;
        end
    end
    h = hits/n_probes; %length(probes(:,1));
    %disp(['measured hit probability: ' num2str(h)])
    h_estimated(bindx) = h;
    % make a decision:
    if abs(h- h_predicted(1,bindx) ) <= abs(h-h_predicted(2,bindx) )
        disp(['detect ' policy ' as: FIFO'])
    else
        disp(['detect ' policy ' as: LRU'])
    end
end
% save(['data/hit_ratio_b_' policy '.mat'],'B','probing_rate','h_ideal','h_predicted','h_estimated');


figure; % probing rate
plot(B, probing_rate, '*k-', 'LineWidth', 1.5);
xlabel('#repetitions b')
ylabel('probing rate (packets/ms)')

figure; % hit ratio
plot(B, h_ideal(1,:), 'm--',... FIFO
    B, h_ideal(2,:), 'c--',... LRU
    B, h_predicted(1,:), 'rs-',... FIFO
    B, h_predicted(2,:), 'bo-',... LRU
    B, h_estimated, 'k*', 'LineWidth', 1.5); 
legend('ideal: FIFO','          LRU','predicted: FIFO','                LRU','measured')
xlabel('#repetitions b')
ylabel('hit ratio')
