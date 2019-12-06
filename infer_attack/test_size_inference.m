q = 3;
inputFileName = ['samples/trace2newhard',int2str(q),'.mat'];
inputTrace=cell2mat(struct2cell(load(inputFileName)));
time_end = inputTrace(1,10000);
% test cache size inference in the lightly loaded scenario:
F = max(inputTrace(2,:))+1;
alpha = 1.0; 
C = 1000; 
lm_background = 10000/time_end; % packets/ms
% Ca = C; % #attack flows
dI = 20; % new rule installation time (ms) (not used)
runs = 20;

%% 1. vary n (#repetitions per experiment)
lm_a = 10*lm_background; % per-flow attack rate (for simulation)
c0 = 2; %1024;
N = [1:10];
time_end = max(inputTrace(1,10000), 2*4*C/lm_a); % duration of bgtraffic to be long enough for a single experiment (sending at most 4C probes) 

C_est = zeros(length(N),runs,2);
Probes = C_est; 
for r=1:runs    
%     disp(' '); disp(['run ' num2str(r) '...'])
%     trace=cell2mat(struct2cell(load(inputFileName)));
%     bgtraffic = trace.';
%    bgtraffic = exp_trace(F, alpha, lm_background, time_end);
    for nindx = 1:length(N)
        n = N(nindx);
        tic
        [C_est(nindx,r,1),Probes(nindx,r,1)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'FIFO');
        [C_est(nindx,r,2),Probes(nindx,r,2)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'LRU');
        disp([ 'n = ' num2str(n) ' takes ' num2str(toc) ' seconds'])
    end
end
save('data/size_n.mat', 'C_est','Probes','N','C','lm_background','lm_a');

err = abs(C_est-C); 
err = mean(err,2).*(100/C); % relative error in %
figure; % error
plot(N, reshape(err(:,1,1),size(N)), 'rs-',...
    N, reshape(err(:,1,2),size(N)), 'bo-', 'LineWidth', 1.5);
xlabel('#repetitions n')
ylabel('relative error (%)')
legend('FIFO', 'LRU')
hold on;
bound = 1-exp(-lm_background*4*(C-1)/lm_a); % prob. of no background during one experimentplot(N, bound.^N.*100,'k:', 'LineWidth', 1.5); 
legend('FIFO', 'LRU', 'bound')
hold off;

cost = mean(Probes,2);
figure; % probing cost
plot(N, reshape(cost(:,1,1),size(N)), 'rs-',...
    N, reshape(cost(:,1,2),size(N)), 'bo-', 'LineWidth', 1.5);
xlabel('#repetitions n')
ylabel('#probes')
legend('FIFO', 'LRU')
%% 2. vary lm_a (probing rate)
Lm_a = 1*lm_background:1*lm_background:10*lm_background; %[5 10 15 20:20:100]; 
c0 = 2;
n = 1;
time_end = max(inputTrace(1,10000), 2*4*C/min(Lm_a));

C_est = zeros(length(Lm_a),runs,2);
Probes = C_est;
for r=1:runs    
    disp(' '); disp(['run ' num2str(r) '...'])
    trace=cell2mat(struct2cell(load(inputFileName)));
    bgtraffic = trace.';
    for nindx = 1:length(Lm_a)
        lm_a = Lm_a(nindx);
        tic
        [C_est(nindx,r,1),Probes(nindx,r,1)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'FIFO');
        [C_est(nindx,r,2),Probes(nindx,r,2)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'LRU');
        disp([ 'lm_a = ' num2str(lm_a) ' takes ' num2str(toc) ' seconds'])
    end
end
save('data/size_rate.mat', 'C_est','Probes','Lm_a','C','lm_background','n');

err = abs(C_est-C); 
err = mean(err,2).*(100/C); % relative error in %
figure; % error
plot(Lm_a, reshape(err(:,1,1),size(Lm_a)), 'rs-',...
    Lm_a, reshape(err(:,1,2),size(Lm_a)), 'bo-', 'LineWidth', 1.5);
ylim([0,30]);
xlabel('probing rate')
ylabel('relative error (%)')
legend('FIFO', 'LRU')
hold on;
bound = (1-exp(-lm_background*4*(C-1)./Lm_a)).^n; % upper bound on prob. of error
plot(Lm_a, bound.*100,'k:', 'LineWidth', 1.5); 
legend('FIFO', 'LRU', 'bound')
hold off;

cost = mean(Probes,2);
figure; % probing cost
plot(Lm_a, reshape(cost(:,1,1),size(Lm_a)), 'rs-',...
    Lm_a, reshape(cost(:,1,2),size(Lm_a)), 'bo-', 'LineWidth', 1.5);
xlabel('probing rate')
ylabel('#probes')
legend('FIFO', 'LRU')




% %% 3. vary c0 (initial guess of cache size)
% lm_a = 10*lm_background; % per-flow attack rate (for simulation)
% C0 = [2 200:200:2000];
% n = 1;
% time_end = max(inputTrace(1,10000), 2*4*C/lm_a);
% 
% C_est = zeros(length(C0),runs,2);
% Probes = C_est;
% for r=1:runs    
%     disp(' '); disp(['run ' num2str(r) '...'])
%     trace=cell2mat(struct2cell(load(inputFileName)));
%     bgtraffic = trace.';
%     for nindx = 1:length(C0)
%         c0 = C0(nindx);
%         tic
%         [C_est(nindx,r,1),Probes(nindx,r,1)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'FIFO');
%         [C_est(nindx,r,2),Probes(nindx,r,2)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'LRU');
%         disp([ 'c0 = ' num2str(c0) ' takes ' num2str(toc) ' seconds'])
%     end
% end
% save('data/size_c0.mat', 'C_est','Probes','C0','C','lm_background','lm_a','n');
% 
% err = abs(C_est-C); 
% err = mean(err,2).*(100/C); % relative error in %
% figure; % error
% plot(C0, reshape(err(:,1,1),size(C0)), 'rs-',...
%     C0, reshape(err(:,1,2),size(C0)), 'bo-', 'LineWidth', 1.5);
% ylim([0,10]);
% xlabel('initial guess of cache size c_0')
% ylabel('relative error (%)')
% legend('FIFO', 'LRU')
% hold on;
% bound = (1-exp(-lm_background*4*(C-1)./lm_a)).^n; % upper bound on prob. of errory
% plot(C0, bound*100*ones(size(C0)),'k:', 'LineWidth', 1.5); 
% legend('FIFO', 'LRU', 'bound')
% hold off;
% 
% cost = mean(Probes,2);
% figure; % probing cost
% plot(C0, reshape(cost(:,1,1),size(C0)), 'rs-',...
%     C0, reshape(cost(:,1,2),size(C0)), 'bo-', 'LineWidth', 1.5);
% xlabel('initial guess of cache size c_0')
% ylabel('#probes')
% legend('FIFO', 'LRU')

