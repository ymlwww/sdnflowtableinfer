for q=3:3
    inputFileName = ['trace2newhard',int2str(q),'.mat'];
    inputTrace=cell2mat(struct2cell(load(inputFileName)));
    time_end = inputTrace(1,10000);
    sim_F = max(inputTrace(2,:))+1;
    F = sim_F-50+randi([0 100],1,1);
%     F = 0.95*sim_F;
    sim_alpha = 1.0; 
    alpha = (0.95+rand*0.1)*sim_alpha; 
    sim_C = 1000; 
    C = (0.95+rand*0.1)*sim_C;
    C = C - mod(C,1);
    sim_lm_background = 10000/time_end;
    lm_background = (0.95+rand*0.1)*sim_lm_background; % packets/ms
    Ca = C; % #attack flows
    lm_a = 0:.01:(0.95+rand*0.1)*1; % per-flow attack rate (for theoretical prediction)
    lm_a_sim = 0:.1:1; % ... (for simulation)
    %time_end = 3.436429978000000e+03; 
    dI = 20; % new rule installation time (20 ms)

    %% 1. evenly distribute attack rate
    H_fifo_t = zeros(size(lm_a)); % theoretical prediction
    H_lru_t = H_fifo_t;
    for i=1:length(lm_a)
        lm_attack = lm_a(i)*ones(1,Ca);
        [H_fifo_t(i), H_lru_t(i)] = hit_ratio_with_attack_theory(C, lm_background, alpha, F, lm_attack, dI );
    end

    H_fifo_s = zeros(size(lm_a_sim)); % simulation
    H_lru_s = H_fifo_s;
    for i=1:length(lm_a_sim)
        tic
        lm_attack = lm_a_sim(i)*ones(1,sim_C);
        [H_fifo_s(i), H_lru_s(i)] = hit_ratio_with_attack_sim(sim_C, sim_lm_background, sim_alpha, sim_F, lm_attack, time_end, dI ,inputFileName);
        disp(['lambda_a = ' num2str(lm_a_sim(i)) ' takes ' num2str(toc) ' seconds'])
    end


    % save('data/dos_equalrate.mat','lm_a','lm_a1','H_fifo_s','H_fifo_t','H_lru_s','H_lru_t');
    save(['data/dos_total_new',int2str(q),'.mat'],'lm_a_sim','lm_a','H_fifo_s','H_fifo_t','H_lru_s','H_lru_t','dI');
    % save('data/dos_equalrate_delay_zoomin.mat','lm_a','lm_a1','H_fifo_s','H_fifo_t','H_lru_s','H_lru_t','dI');
    %%
    % load('data/dos_equalrate.mat');
    % load('data/dos_equalrate_delay.mat');
    % load('data/dos_equalrate_delay_zoomin.mat');
%     h=figure;
%     plot(lm_a, H_fifo_s, 'rs',...
%         lm_a1, H_fifo_t, 'r-',...
%         lm_a, H_lru_s, 'bo',...
%         lm_a1, H_lru_t, 'b--', 'LineWidth', 1.5);
%     legend('FIFO experimental', 'FIFO theoretical', 'LRU experimental', 'LRU theoretical')
%     xlabel('attack traffic rate (per flow)')
%     ylabel('hit ratio for legitimate users')
    % %% zoom in (repeat the above under this setting)
    % lm_a = 0:.02:.2;
    % lm_a1 = 0:.002:.2;

    %% compare with the hit ratio of LFU:
%     H_lfu_t = zeros(size(lm_a1));
%     p = (1./(1:F).^alpha);
%     p = p./sum(p); % Zipf popularity distribution
%     lambda = lm_background*p; % rates of background flows
%     for i=1:length(lm_a1)
%         H_lfu_t(i) = sum(p(lambda >= lm_a1(i)));    
%     end
%     hold on;
%     plot(lm_a1, H_lfu_t, 'k-.', 'LineWidth', 1.5);
%     hold off;
% 
%     % annotate the plot:
%     hbar = H_fifo_t(end); % 0.2469 (dI=0); 0.3267 (dI=20); 0.2535 (dI=.02)
%     [~,I] = min(abs(H_lru_t - hbar)); 
%     hold on;
%     plot([lm_a1(I) lm_a1(end)],[hbar hbar], 'k:', 'LineWidth', 1.5);
%     plot(lm_a1(I)*[1 1], [.0 hbar], 'k:', 'LineWidth', 1.5); % lambda_a1(I)=0.15 (dI=20), 3.7 (dI=.02)
%     hold off;
% 
%     % legend('FIFO experimental', 'FIFO theoretical', 'LRU experimental', 'LRU theoretical')
%     legend('FIFO with correct paramters', 'FIFO with correct paramters', 'LRU with correct paramters', 'LRU with estimated paramters', 'LFU')
    %saveas(h,['dosres',int2str(q),'alpha',num2str(alpha),'eps'],'eps');
    %save(['dosres',int2str(q),'alpha',num2str(alpha),'.mat']);
end