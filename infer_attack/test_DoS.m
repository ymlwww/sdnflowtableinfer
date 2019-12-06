q = 3;
inputFileName = ['samples/trace2newhard',int2str(q),'.mat'];
inputTrace=cell2mat(struct2cell(load(inputFileName)));
time_end = inputTrace(1,10000);
sim_F = max(inputTrace(2,:))+1;
F = max(inputTrace(2,:))+1;
sim_alpha = 1.0; 
alpha = 1.0; 
sim_C = 1000; 
C = 1000;
sim_lm_background = 10000/time_end;
lm_background = 10000/time_end; % packets/ms
Ca = C; % #attack flows
lm_a = 0:.1:1; % per-flow attack rate (for simulation)
lm_a1 = 0:.01:1; % ... (for theoretical prediction) 
dI = 20; % new rule installation time (20 ms)

%% 1. evenly distribute attack rate
H_fifo_t = zeros(size(lm_a1)); % theoretical prediction
H_lru_t = H_fifo_t;
for i=1:length(lm_a1)
    lm_attack = lm_a1(i)*ones(1,Ca);
    [H_fifo_t(i), H_lru_t(i)] = hit_ratio_with_attack_theory(sim_C, sim_lm_background, sim_alpha, sim_F, lm_attack, dI );
end

% H_fifo_t = zeros(size(lm_a)); % simulation
% H_lru_t = H_fifo_t;
% for i=1:length(lm_a)
%     tic
%     lm_attack = lm_a1(i)*ones(1,Ca);
%     [H_fifo_t(i), H_lru_t(i)] = hit_ratio_with_attack_sim(sim_C, sim_lm_background, sim_alpha, sim_F, lm_attack, time_end, dI ,inputFileName);
%     disp(['lambda_a = ' num2str(lm_a1(i)) ' takes ' num2str(toc) ' seconds'])
% end

H_fifo_s = zeros(size(lm_a)); % simulation
H_lru_s = H_fifo_s;
H_qlru_s = H_fifo_s;
H_random_s = H_fifo_s;
H_lru2_s = H_fifo_s;
H_random2_s = H_fifo_s;
H_arc_s = H_fifo_s;
for i=1:length(lm_a)
    tic
    lm_attack = lm_a(i)*ones(1,Ca);
    [H_fifo_s(i), H_lru_s(i), H_qlru_s(i),H_random_s(i),H_lru2_s(i),H_random2_s(i),H_arc_s(i)] = hit_ratio_with_attack_sim(C, lm_background, alpha, F, lm_attack, time_end, dI ,inputFileName);
%     [H_arc_s(i)] = hit_ratio_with_attack_sim(C, lm_background, alpha, F, lm_attack, time_end, dI ,inputFileName);
    disp(['lambda_a = ' num2str(lm_a(i)) ' takes ' num2str(toc) ' seconds'])
end


% save('data/dos_equalrate.mat','lm_a','lm_a1','H_fifo_s','H_fifo_t','H_lru_s','H_lru_t');
% save('data/dos_equalrate_delay.mat','lm_a','lm_a1','H_fifo_s','H_fifo_t','H_lru_s','H_lru_t','dI');
% save('data/dos_equalrate_delay_zoomin.mat','lm_a','lm_a1','H_fifo_s','H_fifo_t','H_lru_s','H_lru_t','dI');
%%
% load('data/dos_equalrate.mat');
% load('data/dos_equalrate_delay.mat');
% load('data/dos_equalrate_delay_zoomin.mat');
h=figure;
plot(lm_a, H_fifo_s, lm_a, H_lru_s, lm_a, H_qlru_s, lm_a, H_random_s, lm_a, H_lru2_s, lm_a, H_random2_s , lm_a, H_arc_s);
% plot(lm_a, H_arc_s);
legend('FIFO', 'LRU', 'qLRU', 'random', 'lru2' , 'random2', 'arc')
xlabel('attack traffic rate (per flow)')
ylabel('hit ratio for legitimate users')
hold on;
% %% zoom in (repeat the above under this setting)
% lm_a = 0:.02:.2;
% lm_a1 = 0:.002:.2;

%% compare with the hit ratio of LFU:
H_lfu_t = zeros(size(lm_a1));
p = (1./(1:F).^alpha);
p = p./sum(p); % Zipf popularity distribution
lambda = lm_background*p; % rates of background flows
for i=1:length(lm_a1)
    H_lfu_t(i) = sum(p(lambda >= lm_a1(i)));    
end
hold on;
plot(lm_a1, H_lfu_t, 'k-.', 'LineWidth', 1.5);
hold off;

% % annotate the plot:
% hbar = H_fifo_t(end); % 0.2469 (dI=0); 0.3267 (dI=20); 0.2535 (dI=.02)
% [~,I] = min(abs(H_lru_t - hbar)); 
% hold on;
% plot([lm_a1(I) lm_a1(end)],[hbar hbar], 'k:', 'LineWidth', 1.5);
% plot(lm_a1(I)*[1 1], [.0 hbar], 'k:', 'LineWidth', 1.5); % lambda_a1(I)=0.15 (dI=20), 3.7 (dI=.02)
% hold off;

% legend('FIFO experimental', 'FIFO theoretical', 'LRU experimental', 'LRU theoretical')

legend('FIFO', 'LRU', 'qLRU', 'random', 'lru2' , 'random2', 'arc', 'LFU')
legend('arc', 'LFU')


% legend('FIFO experimental', 'FIFO theoretical', 'LRU experimental', 'LRU theoretical', 'LFU')
%saveas(h,['dosres',int2str(q),'alpha',num2str(alpha),'eps'],'eps');
%save(['dosres',int2str(q),'alpha',num2str(alpha),'.mat']);