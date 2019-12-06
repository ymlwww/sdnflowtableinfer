for q=1:20
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

%     %% 1. vary n (#repetitions per experiment)
%     lm_a = 100*lm_background; % per-flow attack rate (for simulation)
%     c0 = 2; %1024;
%     N = [1:10];
%     time_end = max(inputTrace(1,10000), 2*4*C/lm_a); % duration of bgtraffic to be long enough for a single experiment (sending at most 4C probes) 
% 
%     C_est = zeros(length(N),runs,2);
%     Probes = C_est; 
%     for r=1:runs    
%         disp(' '); disp(['run ' num2str(r) '...'])
%         trace=cell2mat(struct2cell(load(inputFileName)));
%         bgtraffic = trace.';
%     %    bgtraffic = exp_trace(F, alpha, lm_background, time_end);
%         for nindx = 1:length(N)
%             n = N(nindx);
%             tic
%             [C_est(nindx,r,1),Probes(nindx,r,1)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'FIFO');
%             [C_est(nindx,r,2),Probes(nindx,r,2)] = RCSE(n,c0,C,bgtraffic,lm_a,dI,'LRU');
%             disp([ 'n = ' num2str(n) ' takes ' num2str(toc) ' seconds'])
%         end
%     end
%     save(['data/originalsize_n',int2str(q),'.mat'], 'C_est','Probes','N','C','lm_background','lm_a');
%% 2. vary lm_a (probing rate)
    Lm_a = 10*lm_background:10*lm_background:100*lm_background; %[5 10 15 20:20:100]; 
%     Lm_a = 50*lm_background;%[5 10 15 20:20:100]; 
    % time_end = inputTrace(1,10000);

    C_est = zeros(length(Lm_a),runs,2);
    % Probes = C_est;
    for r=1:runs    

        trace=cell2mat(struct2cell(load(inputFileName)));
        bgtraffic = trace.';
    %     bgtraffic = exp_trace(F, alpha, lm_background, time_end);
        for nindx = 1:length(Lm_a)
            disp(' '); disp(['run ' num2str(q) ' file...'])
            disp(' '); disp(['run ' num2str(r) '...'])
            disp(' '); disp(['run ' num2str(nindx) 'lm...'])
            lm_a = Lm_a(nindx);
            tic
            C_est(nindx,r,1) = oldRCSE(C,bgtraffic,lm_a,dI,'FIFO');
%             C_est(nindx,r,2) = oldRCSE(C,bgtraffic,lm_a,dI,'LRU');
            disp([ 'lm_a = ' num2str(lm_a) ' takes ' num2str(toc) ' seconds'])
        end
    end
    save(['data/fifo_trace_old_size_rate',int2str(q),'.mat'], 'C_est','Lm_a','C','lm_background');
end
