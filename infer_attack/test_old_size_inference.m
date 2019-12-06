% q = 3;
% inputFileName = ['trace2newhard',int2str(q),'.mat'];
% inputTrace=cell2mat(struct2cell(load(inputFileName)));
% time_end = inputTrace(1,10000);
time_end = 10000;
% test cache size inference in the lightly loaded scenario:
% F = max(inputTrace(2,:))+1;
F =5000;
alpha = 1.0; 
C = 1000; 
% lm_background = 10000/time_end; % packets/ms
lm_background = 0.01;
% Ca = C; % #attack flows
dI = 20; % new rule installation time (ms) (not used)
runs = 20;

% % 1. vary n (#repetitions per experiment)
% lm_a = 100*lm_background; % per-flow attack rate (for simulation)
% lm_a = 1;
% N = [1:10];
% % time_end = inputTrace(1,10000);
% C_est = zeros(runs,2);
% for r=1:runs    
% %     disp(' '); disp(['run ' num2str(r) '...'])
% %     trace=cell2mat(struct2cell(load(inputFileName)));
% %     bgtraffic = trace.';
%     bgtraffic = exp_trace(F, alpha, lm_background, time_end);
%     tic
%     C_est(r,1) = oldRCSE(C,bgtraffic,lm_a,dI,'FIFO');
%     C_est(r,2) = oldRCSE(C,bgtraffic,lm_a,dI,'LRU');
%     disp([ 'r = ' num2str(r) ' takes ' num2str(toc) ' seconds'])
% end
% 
% C_est
%% 2. vary lm_a (probing rate)
Lm_a = 10*lm_background:10*lm_background:100*lm_background; %[5 10 15 20:20:100]; 

% time_end = inputTrace(1,10000);

C_est = zeros(length(Lm_a),runs,2);
% Probes = C_est;
for r=1:runs    
     
%     trace=cell2mat(struct2cell(load(inputFileName)));
%     bgtraffic = trace.';
    bgtraffic = exp_trace(F, alpha, lm_background, time_end);
    for nindx = 1:length(Lm_a)
        disp(' '); disp(['run ' num2str(r) '...'])
        lm_a = Lm_a(nindx);
        tic
        C_est(nindx,r,1) = oldRCSE(C,bgtraffic,lm_a,dI,'FIFO');
        C_est(nindx,r,2) = oldRCSE(C,bgtraffic,lm_a,dI,'LRU');
        disp([ 'lm_a = ' num2str(lm_a) ' takes ' num2str(toc) ' seconds'])
    end
end
save('data/old_size_rate.mat', 'C_est','Lm_a','C','lm_background');
% N = 10:10:100;
% err = abs(C_est-C); 
% err = mean(err,2).*(100/C); % relative error in %
% figure; % error
% plot(N, reshape(err(:,1,1),size(Lm_a)), 'rs-',...
%     N, reshape(err(:,1,2),size(Lm_a)), 'bo-', 'LineWidth', 1.5);
% % ylim([0,30]);
% xlabel('probing rate')
% ylabel('relative error (%)')
% legend('FIFO', 'LRU')



% hold on;
% bound = (1-exp(-lm_background*4*(C-1)./Lm_a)).^n; % upper bound on prob. of error
% plot(Lm_a, bound.*100,'k:', 'LineWidth', 1.5); 
% legend('FIFO', 'LRU', 'bound')
% hold off;
% 
% cost = mean(Probes,2);
% figure; % probing cost
% plot(Lm_a, reshape(cost(:,1,1),size(Lm_a)), 'rs-',...
%     Lm_a, reshape(cost(:,1,2),size(Lm_a)), 'bo-', 'LineWidth', 1.5);
% xlabel('probing rate')
% ylabel('#probes')
% legend('FIFO', 'LRU')




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
