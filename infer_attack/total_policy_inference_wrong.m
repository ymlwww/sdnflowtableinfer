for q=1:20
    inputFileName = ['trace2newhardshort',int2str(q),'.mat'];
    inputTrace=cell2mat(struct2cell(load(inputFileName)));
    time_end = inputTrace(1,1000);
    % test size-aware policy inference (RCPD) in the lightly loaded scenario:
    F = max(inputTrace(2,:))+1;
    alpha = 1.0; 
    C = 1000; 
    lm_background = 1000/time_end; % packets/ms
    % Ca = C; % #attack flows
    dI = 0; % new rule installation time (ms) (not used)
    runs = 50;
    index_fifo = 0;
    index_lru = 1;

    %% 1. vary N
    lm_a = 1000*lm_background; %1;
    C_est = 990;
    N = [1:10]';
    time_end = max(time_end, 4*C_est/lm_a); % duration of bgtraffic to be long enough for a single experiment (sending C+3 probes)

    Pe = zeros(length(N),runs,2); 
    Probes = Pe;
    for r=1:runs
        disp(' '); disp(['run ' num2str(r) '...'])
    %     trace=cell2mat(struct2cell(load(inputFileName)));
    %     bgtraffic = trace.';     
        bgtraffic = exp_trace(F, alpha, lm_background, time_end);
        for nindx = 1:length(N)
            n = N(nindx);
            tic
            [p,Probes(nindx,r,1)] = RCPD(C_est,C,n,bgtraffic,lm_a,dI,'FIFO');
            Pe(nindx,r,1) = (p~=index_fifo);
            [p,Probes(nindx,r,2)] = RCPD(C_est,C,n,bgtraffic,lm_a,dI,'LRU');
            Pe(nindx,r,2) = (p~=index_lru);
            disp([ 'n = ' num2str(n) ' takes ' num2str(toc) ' seconds'])
        end
    end
    save(['data/newpolicy_N',int2str(q),'.mat'], 'Pe','Probes','N','lm_background','C','lm_a');    

%     pe = reshape(mean(Pe,2),length(N),2);
% 
%     figure; % error
%     plot(N, pe(:,1), 'rs-',... when ground truth is FIFO
%         N, pe(:,2), 'bo-', 'LineWidth', 1.5); % when ground truth is LRU
%     xlabel('#experiments N')
%     ylabel('error probability')
%     legend('FIFO','LRU')
%     % hold on;
%     % bound = 1-exp(-lm_background*(C+3)/lm_a);
%     % plot(N, bound.^N, 'k:', 'LineWidth', 1.5);
%     % legend('FIFO', 'LRU', 'bound')
%     % hold off;
% 
%     cost = reshape(mean(Probes,2),length(N),2);
%     figure;
%     plot(N, cost(:,1), 'rs-',...
%         N, cost(:,2), 'bo-', 'LineWidth', 1.5);
%     xlabel('#experiments N')
%     ylabel('#probes')
%     legend('FIFO', 'LRU')

%     %% 2. vary lm_a
%     Lm_a = [100*lm_background:1000*lm_background];
%     C_est = C;
%     n = 10;
%     time_end = max(time_end, 4*C_est/min(Lm_a));
% 
%     Pe = zeros(length(Lm_a),runs,2);
%     Probes = Pe;
%     for r=1:runs
%         disp(' '); disp(['run ' num2str(r) '...'])
%         trace=cell2mat(struct2cell(load(inputFileName)));
%         bgtraffic = trace.';
%         for nindx = 1:length(Lm_a)
%             lm_a = Lm_a(nindx);
%             tic
%             [p,Probes(nindx,r,1)] = RCPD(C_est,C,n,bgtraffic,lm_a,dI,'FIFO');
%             Pe(nindx,r,1) = (p~=index_fifo);
%             [p,Probes(nindx,r,2)] = RCPD(C_est,C,n,bgtraffic,lm_a,dI,'LRU');
%             Pe(nindx,r,2) = (p~=index_lru);
%             disp([ 'lm_a = ' num2str(lm_a) ' takes ' num2str(toc) ' seconds'])
%         end
%     end
%     save(['data/newpolicy_rate',int2str(q),'.mat'], 'Pe','Probes','n','lm_background','C','Lm_a');  
%     %%
%     load('data/policy_rate67.mat');
%     pe67 = reshape(mean(Pe,2),length(Lm_a),2); % only for Lm_a = [6 7]
%     cost67 = reshape(mean(Probes,2),length(Lm_a),2);
%     load('data/policy_rate.mat');
%     pe = reshape(mean(Pe,2),length(Lm_a),2); % for Lm_a = 1:10
%     cost = reshape(mean(Probes,2),length(Lm_a),2);
%     pe([6 7],:) = (pe([6 7],:) + pe67)./2;
%     cost([6 7],:) = (cost([6 7],:) + cost67)./2;
% 
%     figure; % error
%     plot(Lm_a, pe(:,1), 'rs-',... when ground truth is FIFO
%         Lm_a, pe(:,2), 'bo-', 'LineWidth', 1.5); % when ground truth is LRU
%     xlabel('probing rate')
%     ylabel('error probability')
%     legend('FIFO','LRU')
%     hold on;
%     bound = 1-exp(-lm_background*(C+3)./Lm_a);
%     plot(Lm_a, bound.^n, 'k:', 'LineWidth', 1.5);
%     legend('FIFO', 'LRU', 'bound')
%     hold off;
% 
%     figure;
%     plot(Lm_a, cost(:,1), 'rs-',...
%         Lm_a, cost(:,2), 'bo-', 'LineWidth', 1.5);
%     xlabel('probing rate')
%     ylabel('#probes')
%     legend('FIFO', 'LRU')

%     %% 3. vary C_est (sensitivity to errors in estimating the cache size)
%     lm_a = 1000*lm_background;
%     C_est_total = [C-4:C];
%     n = 10;
%     time_end = max(time_end, 4*C/lm_a);
% 
%     Pe = zeros(length(C_est_total),runs,2);
%     Probes = Pe;
%     for r=1:runs
%         disp(' '); disp(['run ' num2str(r) '...'])
%         trace=cell2mat(struct2cell(load(inputFileName)));
%         bgtraffic = trace.';
%         for nindx = 1:length(C_est_total)
%             C_est = C_est_total(nindx);
%             tic
%             [p,Probes(nindx,r,1)] = RCPD(C_est,C,n,bgtraffic,lm_a,dI,'FIFO');
%             Pe(nindx,r,1) = (p~=index_fifo);
%             [p,Probes(nindx,r,2)] = RCPD(C_est,C,n,bgtraffic,lm_a,dI,'LRU');
%             Pe(nindx,r,2) = (p~=index_lru);
%             disp([ 'C_est = ' num2str(C_est) ' takes ' num2str(toc) ' seconds'])
%         end
%     end
%     save(['data/policy_Cest',int2str(q),'.mat'], 'Pe','Probes','n','lm_background','C','lm_a','C_est_total');  
% 
% %     pe = reshape(mean(Pe,2),length(C_est_total),2);
% %     figure; % error
% %     plot(C_est_total, pe(:,1), 'rs-',... when ground truth is FIFO
% %         C_est_total, pe(:,2), 'bo-', 'LineWidth', 1.5); % when ground truth is LRU
% %     xlabel('estimated cache size')
% %     ylabel('error probability')
% %     legend('FIFO','LRU')
% %     % hold on;
% %     % bound = 1-exp(-lm_background*(C+3)./Lm_a);
% %     % plot(Lm_a, bound.^n, 'k:', 'LineWidth', 1.5);
% %     % legend('FIFO', 'LRU', 'bound')
% %     % hold off;
% % 
% %     cost = reshape(mean(Probes,2),length(C_est_total),2);
% %     figure;
% %     plot(C_est_total, cost(:,1), 'rs-',...
% %         C_est_total, cost(:,2), 'bo-', 'LineWidth', 1.5);
% %     xlabel('estimated cache size')
% %     ylabel('#probes')
% %     legend('FIFO', 'LRU')
% 
%     %% find out how much estimation there can be:
%     lm_a = 1000*lm_background; % per-flow attack rate (for simulation)
%     c0 = 2; %1024;
%     n = 1;
%     time_end = max(time_end, 2*4*C/lm_a); % duration of bgtraffic to be long enough for a single experiment (sending at most 4C probes) 
%     runs1 = 50;
% 
%     C_est = zeros(runs1,2);
%     Probes = C_est; 
%     for r=1:runs1    
%         disp(' '); disp(['run ' num2str(r) '...'])
%         trace=cell2mat(struct2cell(load(inputFileName)));
%         bgtraffic = trace.';
%         tic
%         [C_est(r,1),Probes(r,1)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'FIFO');
%         [C_est(r,2),Probes(r,2)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'LRU');
%         disp([ 'takes ' num2str(toc) ' seconds'])
%     end
%     save(['data/policy_last',int2str(q),'.mat'], 'Pe','Probes','n','lm_background','C','lm_a','C_est_total');  
% %     figure;
% %     plot(1:runs1, C*ones(1,runs1), 'k--',...
% %         1:runs1, sort(C_est(:,1)), 'rs-',...
% %         1:runs1, sort(C_est(:,2)), 'bo-', 'LineWidth', 1.5);
% %     xlabel('Monte Carlo runs')
% %     ylabel('estimated cache size')
% %     legend('true value', 'FIFO', 'LRU')
end

