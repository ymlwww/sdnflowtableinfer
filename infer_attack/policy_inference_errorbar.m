pointnum = 10;
runs = 20;
error_fifo=zeros(pointnum,runs);
error_lru=zeros(pointnum,runs);
% Probes_fifo = zeros(pointnum,runs);
% Probes_lru = zeros(pointnum,runs);
for q =1:runs
    inputFileName = ['data/newpolicy_N',int2str(q),'.mat'];
    load(inputFileName);
    pe = reshape(mean(Pe,2),pointnum,2);
    cost = reshape(mean(Probes,2),pointnum,2);
% %     %     cost = reshape(mean(Probes,2),length(N),2);
%     Probes_fifo(:, q) = abs(cost(:,1));
%     Probes_lru(:, q) = abs(cost(:,2));
    error_fifo(:, q) = pe(:,1);
    error_lru(:, q) = pe(:,2);
end
error_fifo_mean = mean(error_fifo,2);
error_lru_mean = mean(error_lru,2);
error_fifo_var = var(error_fifo,0,2);
error_lru_var = var(error_lru,0,2);
figure;
N = 100:100:1000;
errorbar(N, error_fifo_mean, error_fifo_var,'rs-');
hold on;
errorbar(N, error_lru_mean, error_lru_var,'bo-', 'LineWidth', 1.5);
xlabel('probing rate')
ylabel('relative error (%)');
legend('FIFO', 'LRU')
hold off;
% error_fifo_mean = mean(Probes_fifo,2);
% error_lru_mean = mean(Probes_lru,2);
% error_fifo_var = var(Probes_fifo,0,2);
% error_lru_var = var(Probes_lru,0,2);
% N = 1:1:10;
% figure; % probing cost
% errorbar(N, error_fifo_mean, error_fifo_var,'rs-');
% hold on;
% errorbar(N, error_lru_mean, error_lru_var,'bo-', 'LineWidth', 1.5);
% xlabel('probing rate')
% ylabel('#probes')
% legend('FIFO', 'LRU');

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
% 
