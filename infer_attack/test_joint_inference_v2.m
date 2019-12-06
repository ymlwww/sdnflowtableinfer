% Policy-aware cache size & user parameter inference (under heavy load):

%% set the parameters:
C = 1000; % cache size
F = 5000; % number of distinct contents requested by background traffic (i.e., #background flows)
alpha = 0.9; % Zipf parameter of the content popularity distribution (i.e., flow size distribution)
lambda_background = 10; % background traffic rate (packets/ms)
lambda_a = 1; % maximum probing rate 
n_flows_a = 100; % maximum #probing flows
% Cinit = ones(C,1)*(-2); % initial cache state: empty cache
Cinit = [0:C-1]'; % most popular contents
dI = 20; % average time to get a new content after a miss (i.e., time to install a new rule) (ms)
p = (1./(1:F).^alpha);
%disp(lambda);
Lambda_background = lambda_background*p./sum(p); % Lambda(i): rate of the i-th largest background flow
Fmin = 1000; Fstep = 1000; Fmax = 10000; % parameter for grid search
lambdamin = 1; lambdastep = 1; lambdamax = 10;
alphamin = 0; alphastep = .1; alphamax = 1;
n_eqs = 4; %[2 4 6 8]; % #probing experiments (at least 4)

b = 10; % #repetitions in estimating characteristic time in Method 2
% time_end = 100000; % duration of each probing experiment (ms)
time_end = 1000; % duration of each probing experiment (ms)
runs = 10; % #runs for background traffic generation
debug = 0; % 0: not debug (save results); 1: debug (not save results)

%% run probing experiments, vary time_end (duration of each probing experiment)
% policy = 'FIFO'; policy_wrong = 'LRU';
policy = 'LRU'; policy_wrong = 'FIFO';
N_flows_a = [50 100 500]; %[30 50 100 200 500]; %[50 100 300 500];

for findx = 1:length(N_flows_a)
    n_flows_a = N_flows_a(findx); % maximum #probing flows
fprintf('\n\n %s: n_flows_a = %d\n',policy,n_flows_a)

% design #probing flows in each experiment:
% N_flows = [round(n_flows_a/10) round(n_flows_a/3) round(n_flows_a*2/3) n_flows_a]; design = 'varyflows'; % or starting from round(n_flows_a/10)? [3 round(n_flows_a/3) round(n_flows_a*2/3) n_flows_a];
N_flows = n_flows_a*ones(1,n_eqs); design = 'flows';

Lambda = zeros(1,n_eqs); % Lambda(eindx): probing rate for the eindx-th experiment
Tau_true = zeros(1,n_eqs);
Hit_true = Tau_true;
tau0 = 0;
[ tau_bg ] = characteristic_time( Lambda_background,dI,C,tau0,policy ); 
switch policy
    case 'FIFO'
%         hit_target = [.1:(.9-.1)/(n_eqs-1):.9]; %.9*ones(1,4);
%         hit_target = [.9:-(.9-.1)/(n_eqs-1):.1];
          high = .92; low = .3;
    case 'LRU'
%         hit_target = [.1:(.99-.1)/(n_eqs-1):.99]; %.99*ones(1,4); %.2:(.9-.2)/(n_eqs-1):.9; % .9 for FIFO, .99 for LRU
%         hit_target = [.99:-(.99-.1)/(n_eqs-1):.1];
          high = .99; low = .2; %100/n_flows_a; 
    otherwise
end% hit_target = .9 for FIFO, .99 for LRU %%%%.9:-(.9-.2)/(n_eqs-1):.2;
hit_target = [low:(high-low)/(n_eqs-1):high];
for eindx = 1:n_eqs
    Lambda(eindx) = design_probing_rate(hit_target(eindx), tau_bg, dI, policy);
    Tau_true(eindx) = characteristic_time( [Lambda_background Lambda(eindx)*ones(1,N_flows(eindx))],dI,C,tau0,policy ); % compute true characteristic time    
    Hit_true(eindx) = hit_ratio_TTL(Lambda(eindx), Tau_true(eindx), dI, policy);
end
N_flows
Lambda
Tau_true
Hit_true


Time_end = time_end; % 10000:2000:20000;

t_EST = zeros(2,runs,length(Time_end),n_eqs); % estimated characteristic time
h_EST = t_EST; % hit ratio (per probing flow) computed from the estimated characteristic time
e_thresh = 10^(-4); 
for nindx = 1:length(Time_end)
    time_end = Time_end(nindx); % to estimate F, alpha, lambda_background, C, we need 4 equations at least    
    disp(' '); disp(['time_end = ' num2str(time_end) ':'])
    for rindx = 1:runs % #successful probing rounds
    rindx1 = 1;
    tic
    while 1         
        disp(['time_end = ' num2str(time_end) ', run ' num2str(rindx) ': trial ' num2str(rindx1)])        
        trace = exp_trace(F, alpha, lambda_background, time_end); % background traffic
        Tau = zeros(2,n_eqs);
        Hit = Tau;
        abort1 = 0; abort2 = 0;
        for eindx = 1:n_eqs
            disp(['send ' num2str(N_flows(eindx)) ' flows, each at rate ' num2str(Lambda(eindx)) ])
            %         [ tau ] = characteristic_time( [Lambda_background Lambda(eindx)],dI,C,tau0,policy ); % compute true characteristic time
            tau = Tau_true(eindx);
            probes = exp_trace(1, alpha, Lambda(eindx)*N_flows(eindx), time_end);
            probes(:,2) = F + randi(N_flows(eindx), size(probes(:,2))); % probing packets with IDs randomly selected from F+1,...,F+N_flows(eindx)
            mergetrace = cat(1,trace,probes);
            mergetrace = sortrows(mergetrace,1);
            % Method 1: measure hit ratio, calculate characteristic time
            cache.C = Cinit;
            cache.pend = zeros(0,2);
            hits = zeros(1,N_flows(eindx));
            for i=1:length(mergetrace(:,1))
                [cache, ishit] = CacheAdd_withdelay( cache, mergetrace(i,1), mergetrace(i,2), dI, policy);
                if (mergetrace(i,2) > F)
                    hits(mergetrace(i,2)-F) = hits(mergetrace(i,2)-F) + ishit;
                end
            end           
            % average hit ratio over all probing flows:
            is_valid = zeros(size(hits));
            for i=1:length(hits)
                hits(i) = hits(i)/sum(probes(:,2)==F+i);
                if ~isnan(hits(i))
                    is_valid(i) = 1;
                end
            end
            Hit(1,eindx) = mean(hits(is_valid~=0)); % hits(1); % 
            % aggregate hit ratio for sum of probing flows:
%             Hit(1,eindx) = sum(hits)/sum(probes(:,2)>F);
%             if eindx > 1 && (Hit(1,eindx) <= Hit(1,eindx-1) )
%                 abort1 = 1; 
%                 disp(['abort1 = 1 due to:'])
%                 Hit(1,:)
%                 break;
%             end                      
            Tau(1,eindx) = characteristic_time_hit_ratio( Hit(1,eindx), Lambda(eindx), dI, policy );
%             disp(['with ' num2str(N_flows(eindx)) ' probing flows:'])
%             disp(['method 1: hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ' (true: ' num2str(tau) ')'])
            % Method 2: estimate characteristic time, calculate hit ratio
%             [ T ] = characteristic_time_estimation( mergetrace, tau+10, b, tau*.01, Cinit, dI, policy );
% %             if eindx > 1 && T >= Tau(2,eindx-1) %|| T < Tau(2,eindx-1)-10*(Tau_true(eindx-1)-Tau_true(eindx))) % characteristic time should be decreasing as probing rate increases
% %                 abort2 = 1;
% %                 disp('abort2 = 1 due to:')
% %                 Tau(2,eindx) = T;
% %                 Tau(2,:)
% %                 break;
% %             end
%             Tau(2,eindx) = T;
%             Hit(2,eindx) = hit_ratio_TTL(Lambda(eindx), T, dI, policy); % calculated hit ratio
            %             Hit(2,eindx) = hits/size(probes,1); % measured hit ratio
            %             disp(['method 2: hit probability = ' num2str(Hit_EST(2,rindx,nindx,eindx)) ', tau = ' num2str(T) ' (true: ' num2str(tau) ')'])
        end
        if abort1 || abort2            
            disp(['aborted: abort1 = ' num2str(abort1) ', abort2 = ' num2str(abort2)]); 
            disp(' ')
        else% if the probing experiments complete successfully
            disp(['run ' num2str(rindx) ' takes ' num2str(toc) ' seconds'])
            % print the result:
            disp(' ')
            for eindx = 1:n_eqs
                disp(['at ' num2str(N_flows(eindx)) ' probing flows:'])
                disp(['method 1:   hit probability = ' num2str(Hit(1,eindx)) ', tau = ' num2str(Tau(1,eindx)) ])
%                 disp(['method 2:   hit probability = ' num2str(Hit(2,eindx)) ', tau = ' num2str(Tau(2,eindx)) ])
                disp(['true value: hit probability = ' num2str(Hit_true(eindx)) ', tau = ' num2str(Tau_true(eindx)) ])
            end
            disp(' ')
            %         % solve equations:
            %         disp(' ')
            %         disp('Method 2: estimate characteristic time, calculate hit ratio')
            %         [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference( Hit(2,1,:), Tau(2,1,:), Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy );
            %         if error < e_thresh % successfully approximate the equations
            %             c_EST(rindx,2,nindx) = C_est;
            %             f_EST(rindx,2,nindx) = F_est;
            %             lm_EST(rindx,2,nindx) = lambda_est;
            %             a_EST(rindx,2,nindx) = alpha_est;
            for i=1:2
                t_EST(i,rindx,nindx,:) = reshape(Tau(i,:),1,1,1,n_eqs);
                h_EST(i,rindx,nindx,:) = reshape(Hit(i,:),1,1,1,n_eqs);
            end
            break;
            %         end
        end
        rindx1 = rindx1 + 1;        
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
end% for nindx = 1:length(Time_end)
if ~debug
    save(['data/joint_inference_' policy '_' num2str(n_flows_a) design '.mat'], 'Time_end', 'N_flows', 'Lambda', 'Hit_true','Tau_true', 'h_EST', 't_EST'); %, 'c_est','f_est','lm_est','a_est','C','F','lambda_background','alpha');
end
end %for findx = 1:length(N_flows_a)

% %% concatenating added Monte Carlo runs with previous runs:
% for findx = 1:length(N_flows_a)
%     n_flows_a = N_flows_a(findx);
%     load(['data/joint_inference_' policy '_' num2str(n_flows_a) design '.mat']);
%     h_EST1 = h_EST;
%     t_EST1 = t_EST;
%     load(['data/joint_inference_' policy '_' num2str(n_flows_a) design '_Copy.mat']);
%     h_EST = cat(2,h_EST1,h_EST); % new, old
%     t_EST = cat(2,t_EST1,t_EST);
%     save(['data/joint_inference_' policy '_' num2str(n_flows_a) design '.mat'], 'Time_end', 'N_flows', 'Lambda', 'Hit_true','Tau_true', 'h_EST', 't_EST');
% end

%% after probing experiment, solve the equations to infer lambda, F, alpha for given C: (Fig. 7-8 (a-c))
% policy = 'FIFO'; policy_wrong = 'LRU';
policy = 'LRU'; policy_wrong = 'FIFO';
N_flows_a = [50 100 200 300 500]; %[50 100 300 500];

% infer lambda, F, alpha for given C:
trueC = C; % if > 0, assumed to be the true value
trueF = 0; % if > 0, assumed to be the true value
trueLambda = 0; % if > 0, assumed to be the true value
trueAlpha = -1; % if >= 0, assumed to be the true value

% load(['data/joint_inference_' policy '_2round.mat']); % if only rerun for certain #flows
C_EST1 = zeros(length(N_flows_a),2,runs); 
L_EST1 = zeros(length(N_flows_a),2,runs);
F_EST1 = zeros(length(N_flows_a),2,runs);
A_EST1 = zeros(length(N_flows_a),2,runs);
xlabels = cell(size(N_flows_a));

for j=1:length(N_flows_a)
    n_flows_a = N_flows_a(j);
%     load(['data/joint_inference_' policy '_' num2str(n_flows_a) 'varyflows.mat']);
     load(['data/joint_inference_' policy '_' num2str(n_flows_a) 'flows.mat']);
    % disp(' ')
    % disp('Ideal: both hit ratio and characteristic time are calculated')
    [ C_est_ideal, F_est_ideal, lambda_est_ideal, alpha_est_ideal, error ] = joint_parameter_inference_new( Hit_true, Tau_true, N_flows, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy, 0,0,0,-1 );
    % Note: case "ideal" always yield perfect estimates, so skipped.
for r=1:size(t_EST,2) % run r:
    for i=1%:2
        disp(' ')
        disp(['true policy, n_flows_a = ' num2str(n_flows_a) ', method ' num2str(i) ', run ' num2str(r) ':'])
        hit_est = h_EST(i,r,end,2:4);
        tau_est = t_EST(i,r,end,2:4);        
        [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference_new( hit_est, tau_est, N_flows, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy, trueC, trueF, trueLambda, trueAlpha );
        C_EST1(j,i,r) = C_est;
        F_EST1(j,i,r) = F_est;
        L_EST1(j,i,r) = lambda_est;
        A_EST1(j,i,r) = alpha_est;        
        disp(' ')
        disp(['wrong policy, n_flows_a = ' num2str(n_flows_a) ', method ' num2str(i) ', run ' num2str(r) ':'])
        [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference_new( hit_est, tau_est, N_flows, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy_wrong, trueC, trueF, trueLambda, trueAlpha );
        C_EST1(j,i+1,r) = C_est;
        F_EST1(j,i+1,r) = F_est;
        L_EST1(j,i+1,r) = lambda_est;
        A_EST1(j,i+1,r) = alpha_est;  
    end
end
xlabels{j} = num2str(n_flows_a);
end

%% save data if satisfactory:
if ~debug
    save(['data/joint_inference_' policy '_2round.mat'], 'Time_end', 'N_flows_a', 'C_est_ideal','F_est_ideal','lambda_est_ideal','alpha_est_ideal',...
        'C_EST1','F_EST1','L_EST1','A_EST1','C_EST2','F_EST2','L_EST2','A_EST2','C','F','lambda_background','alpha','xlabels');
end

%% plot results: 
% if 'FIFO':
load(['data/joint_inference_' policy '_2round.mat']);
settings = [1 2 3 4]; % plot cases N_flows_a(settings)
N_flows_a = N_flows_a(settings);
C_EST1_1 = zeros(length(N_flows_a),2,runs);
L_EST1_1 = zeros(length(N_flows_a),2,runs);
F_EST1_1 = zeros(length(N_flows_a),2,runs);
A_EST1_1 = zeros(length(N_flows_a),2,runs);
C_EST1_1([1 2],:,:) = C_EST1([1 2],:,1:runs); % use new data
L_EST1_1([1 2],:,:) = L_EST1([1 2],:,1:runs); 
F_EST1_1([1 2],:,:) = F_EST1([1 2],:,1:runs); 
A_EST1_1([1 2],:,:) = A_EST1([1 2],:,1:runs);
C_EST1_1([3 4],:,:) = C_EST1([3 4],:,runs+1:2*runs); % use old data
L_EST1_1([3 4],:,:) = L_EST1([3 4],:,runs+1:2*runs); 
F_EST1_1([3 4],:,:) = F_EST1([3 4],:,runs+1:2*runs); 
A_EST1_1([3 4],:,:) = A_EST1([3 4],:,runs+1:2*runs);
C_EST1 = C_EST1_1;
L_EST1 = L_EST1_1;
F_EST1 = F_EST1_1;
A_EST1 = A_EST1_1;
xlabels = xlabels(settings);
%% if 'LRU':
load(['data/joint_inference_' policy '_2round.mat']);
settings = [1 2 4 5]; % plot cases N_flows_a(settings)
N_flows_a = N_flows_a(settings);
C_EST1 = C_EST1(settings,:,1:runs);
L_EST1 = L_EST1(settings,:,1:runs);
F_EST1 = F_EST1(settings,:,1:runs);
A_EST1 = A_EST1(settings,:,1:runs);
xlabels = xlabels(settings);

%%
labelsize = 16;
legendsize = 14;

figure;
error1 = 100.*abs(L_EST1-lambda_background)/lambda_background;
y1 = reshape(mean(error1,3), length(N_flows_a),2);
% error2 = 100.*abs(L_EST2-lambda_background)/lambda_background;
% y2 = reshape(mean(error2,3), length(N_flows_a),2);
% error_ideal = 100.*abs(lambda_est_ideal-lambda_background)/lambda_background;
bar(y1); % only plot round 1
% bar([y1 y2]); % plot round 1 and 2
xticklabels(xlabels); 
h = legend('true policy','wrong policy');
% h = legend('round 1: true policy','           wrong policy',...
%     'round 2: true policy','           wrong policy');%,...
set(h,'FontSize',legendsize);
%     'ideal:       true policy','           wrong policy')
xlabel('no. probing flows','FontSize',labelsize)
ylabel('relative error (%)','FontSize',labelsize)
% ylim([-1 max(max(max(y1)) )+1])
% ylabel('background traffic rate \lambda')

figure;
error1 = 100.*abs(F_EST1-F)/F;
y1 = reshape(mean(error1,3), length(N_flows_a),2);
% error2 = 100.*abs(F_EST2-F)/F;
% y2 = reshape(mean(error2,3), length(N_flows_a),2);
% error_ideal = 100.*abs(F_est_ideal-F)/F;
bar(y1); % only plot round 1
% bar([y1 y2]); % plot round 1 and 2
xticklabels(xlabels); 
h = legend('true policy','wrong policy');
% h = legend('round 1: true policy','           wrong policy',...
%     'round 2: true policy','           wrong policy');%,...
set(h,'FontSize',legendsize);
%     'ideal:       true policy','           wrong policy')
xlabel('no. probing flows','FontSize',labelsize)
ylabel('relative error (%)','FontSize',labelsize)
% ylabel('no. background flows F')

figure;
error1 = 100.*abs(A_EST1-alpha)/alpha;
y1 = reshape(mean(error1,3), length(N_flows_a),2);
% error2 = 100.*abs(A_EST2-alpha)/alpha;
% y2 = reshape(mean(error2,3), length(N_flows_a),2);
% error_ideal = 100.*abs(alpha_est_ideal-alpha)/alpha;
bar(y1); % only plot round 1
% bar([y1 y2]); % plot round 1 and 2
xticklabels(xlabels); 
h = legend('true policy','wrong policy');
% h = legend('round 1: true policy','           wrong policy',...
%     'round 2: true policy','           wrong policy');%,...
set(h,'FontSize',legendsize);
%     'ideal:       true policy','           wrong policy')
xlabel('no. probing flows','FontSize',labelsize)
ylabel('relative error (%)','FontSize',labelsize)
% ylabel('background traffic skewness \alpha')


%% solve equations for estimating C (jointly with lambda, F, alpha): (Fig. 7-8 (d))
% policy = 'FIFO'; policy_wrong = 'LRU';
policy = 'LRU'; policy_wrong = 'FIFO';

design = 'flows';
N_flows_a =  [50 100 300 500];
trueC = 0; % if > 0, assumed to be the true value
trueF = 0; % if > 0, assumed to be the true value
trueLambda = 0; % if > 0, assumed to be the true value
trueAlpha = -1; % if >= 0, assumed to be the true value
load(['data/joint_inference_' policy '_' num2str(N_flows_a(1)) design '.mat']); % to get #Monte Carlo runs size(t_EST,2)
C_EST = zeros(length(N_flows_a),2,size(t_EST,2)); 
L_EST = zeros(length(N_flows_a),2,size(t_EST,2));
F_EST = zeros(length(N_flows_a),2,size(t_EST,2));
A_EST = zeros(length(N_flows_a),2,size(t_EST,2));
xlabels = cell(size(N_flows_a));
for j=1:length(N_flows_a)
    n_flows_a = N_flows_a(j);
    load(['data/joint_inference_' policy '_' num2str(n_flows_a) design '.mat']);
%     load(['data/joint_inference_' policy '_' num2str(n_flows_a) 'varyflows.mat']);
    
    % disp(' ')
    % disp('Ideal: both hit ratio and characteristic time are calculated')
    % [ C_est_ideal, F_est_ideal, lambda_est_ideal, alpha_est_ideal, error ] = joint_parameter_inference_new( Hit_true, Tau_true, N_flows, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy, 0,0,0,-1 );
    % Note: case "ideal" always yield perfect estimates, so skipped.
for r=1:size(t_EST,2) % run r:
    for i=1%:2
        disp(' ')
        disp(['true policy, n_flows_a = ' num2str(n_flows_a) ', method ' num2str(i) ', run ' num2str(r) ':'])
        hit_est = h_EST(i,r,end,:);
        tau_est = t_EST(i,r,end,:);        
        [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference_new( hit_est, tau_est, N_flows, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy, trueC, trueF, trueLambda, trueAlpha );
        C_EST(j,i,r) = C_est;
        F_EST(j,i,r) = F_est;
        L_EST(j,i,r) = lambda_est;
        A_EST(j,i,r) = alpha_est;        
        disp(' ')
        disp(['wrong policy, n_flows_a = ' num2str(n_flows_a) ', method ' num2str(i) ', run ' num2str(r) ':'])
        [ C_est, F_est, lambda_est, alpha_est, error ] = joint_parameter_inference_new( hit_est, tau_est, N_flows, Fmin,Fstep,Fmax,lambdamin,lambdastep,lambdamax,alphamin,alphastep,alphamax, dI, policy_wrong, trueC, trueF, trueLambda, trueAlpha );
        C_EST(j,i+1,r) = C_est;
        F_EST(j,i+1,r) = F_est;
        L_EST(j,i+1,r) = lambda_est;
        A_EST(j,i+1,r) = alpha_est;  
    end
end
xlabels{j} = num2str(n_flows_a);
end

% save results if good:
if ~debug
    save(['data/joint_inference_' policy '_unknownC.mat'], 'N_flows_a', 'C_EST','F_EST','L_EST','A_EST','C','F','lambda_background','alpha');
end

% plot results:
figure;
error = 100.*abs(C_EST-C)/C; % relative error (%)
y = reshape(mean(error,3), length(N_flows_a),2); 
bb = bar(y); 
% set(bb(1),'FaceColor',[0 0 .5]);
% set(bb(2),'FaceColor',[0 .5 .9]);
xticklabels(xlabels); 
% legend('ideal','method 1','method 2')
h = legend('true policy','wrong policy');
set(h,'FontSize',legendsize);
xlabel('no. probing flows','FontSize',labelsize)
% y = y';
% plot(N_flows_a, y(1,:), 'bo-',... method 1, true policy
%     N_flows_a, y(3,:), 'ro-',... method 1, wrong policy
%     N_flows_a, y(2,:), 'cx-',... method 2, true pollicy
%     N_flows_a, y(4,:), 'mx-',... method 2, wrong policy
%     N_flows_a, y(5,:), 'k-',... ideal, true policy
%     N_flows_a, y(6,:), 'k--', 'LineWidth', 1.5); % ideal, wrong policy
% legend('method 1: true policy', '             wrong policy', 'method 2: true policy', '             wrong policy', 'ideal: true policy', '             wrong policy')
% xlabel('max no. probing flows')
ylabel('relative error (%)','FontSize',labelsize)
% ylabel('cache size C')
% %% this plots the error for the rest of the parameters (lambda, F, alpha); not used
% figure;
% error = 100.*abs(L_EST-lambda_background)/lambda_background; % relative error (%)
% y = reshape(mean(error,3), length(N_flows_a),6); 
% bar(y(:,[5 1 2]));
% xticklabels(xlabels); 
% legend('ideal','method 1','method 2')
% xlabel('no. probing flows')
% % y = y';
% % plot(N_flows_a, y(1,:), 'bo-',... method 1, true policy
% %     N_flows_a, y(3,:), 'ro-',... method 1, wrong policy
% %     N_flows_a, y(2,:), 'cx-',... method 2, true pollicy
% %     N_flows_a, y(4,:), 'mx-',... method 2, wrong policy
% %     N_flows_a, y(5,:), 'k-',... ideal, true policy
% %     N_flows_a, y(6,:), 'k--', 'LineWidth', 1.5); % ideal, wrong policy
% % legend('method 1: true policy', '             wrong policy', 'method 2: true policy', '             wrong policy', 'ideal: true policy', '             wrong policy')
% % xlabel('max no. probing flows')
% ylabel('relative error (%)')
% % ylabel('background traffic rate \lambda')
% 
% figure;
% error = 100.*abs(F_EST-F)/F; % relative error (%)
% y = reshape(mean(error,3), length(N_flows_a),6); 
% bar(y(:,[5 1 2]));
% xticklabels(xlabels); 
% legend('ideal','method 1','method 2')
% xlabel('no. probing flows')
% % y = y';
% % plot(N_flows_a, y(1,:), 'bo-',... method 1, true policy
% %     N_flows_a, y(3,:), 'ro-',... method 1, wrong policy
% %     N_flows_a, y(2,:), 'cx-',... method 2, true pollicy
% %     N_flows_a, y(4,:), 'mx-',... method 2, wrong policy
% %     N_flows_a, y(5,:), 'k-',... ideal, true policy
% %     N_flows_a, y(6,:), 'k--', 'LineWidth', 1.5); % ideal, wrong policy
% % legend('method 1: true policy', '             wrong policy', 'method 2: true policy', '             wrong policy', 'ideal: true policy', '             wrong policy')
% % xlabel('max no. probing flows')
% ylabel('relative error (%)')
% % ylabel('no. background flows F')
% 
% figure;
% error = 100.*abs(A_EST-alpha)/alpha; % relative error (%)
% y = reshape(mean(error,3), length(N_flows_a),6); 
% bar(y(:,[5 1 2]));
% xticklabels(xlabels); 
% legend('ideal','method 1','method 2')
% xlabel('no. probing flows')
% % y = y';
% % plot(N_flows_a, y(1,:), 'bo-',... method 1, true policy
% %     N_flows_a, y(3,:), 'ro-',... method 1, wrong policy
% %     N_flows_a, y(2,:), 'cx-',... method 2, true pollicy
% %     N_flows_a, y(4,:), 'mx-',... method 2, wrong policy
% %     N_flows_a, y(5,:), 'k-',... ideal, true policy
% %     N_flows_a, y(6,:), 'k--', 'LineWidth', 1.5); % ideal, wrong policy
% % legend('method 1: true policy', '             wrong policy', 'method 2: true policy', '             wrong policy', 'ideal: true policy', '             wrong policy')
% % xlabel('max no. probing flows')
% ylabel('relative error (%)')
% % ylabel('background traffic skewness \alpha')

