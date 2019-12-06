pointnum = 10;
runs = 20;
error_fifo=zeros(pointnum,runs);
error_lru=zeros(pointnum,runs);
% Probes_fifo = zeros(pointnum,runs);
% Probes_lru = zeros(pointnum,runs);
for q =1:runs
%         inputFileName = ['data/originalsize_n',int2str(q),'.mat'];
%         load(inputFileName);
% %     cost = mean(Probes,2);
        load(['data/trace_old_size_rate',int2str(q),'.mat']);
%         load('data/originalsize_rate.mat');
        err = abs(C_est-C);
        err = mean(err,2).*(100/C); % relative error in %
        error_fifo = err(:,:,1);
        error_lru = err(:,:,2);
%         load(['data/fifo_trace_old_size_rate',int2str(q),'.mat']);
%         err = abs(C_est-C);
%         err = mean(err,2).*(100/C); % relative error in %
%         error_fifo = err(:,:,1);
% %     Probes_fifo(:, q) = cost(:,1,1);
% %     Probes_lru(:, q) = cost(:,1,2);
% 
end
error_fifo_mean = mean(error_fifo,2);
error_lru_mean = mean(error_lru,2);
error_fifo_var = var(error_fifo,0,2);
error_lru_var = var(error_lru,0,2);
figure;
N = 1:1:10;
errorbar(N, error_fifo_mean, error_fifo_var,'rs-');
% plot(N, error_fifo_mean,N,error_lru_mean );
hold on;
errorbar(N, error_lru_mean, error_lru_var,'bo-', 'LineWidth', 1.5);
xlabel('probing rate');
ylabel('relative error (%)');

labelsize = 16;
legendsize = 14;

load('../ting/data/size_rate.mat');
err = abs(C_est-C); 
err = mean(err,2).*(100/C); % relative error in %
% plot(N, reshape(err(:,1,1),size(N)), 'rs-',...
%     N, reshape(err(:,1,2),size(N)), 'bo-', 'LineWidth', 1.5);
errors = 100.*abs(C_est-C)/C; 
errstd = reshape(std(errors,0,2),length(N),2); errstd = errstd';
errorbar(N,reshape(err(:,1,1),size(N)),errstd(1,:), 'ms-','LineWidth', 1.5);
hold on;
errorbar(N,reshape(err(:,1,2),size(N)),errstd(2,:), 'co-', 'LineWidth', 1.5);
hold off;
xlabel('#repetitions n','FontSize',labelsize)
ylabel('relative error (%)','FontSize',labelsize)
h = legend( 'policy-aware: FIFO', 'policy-aware: LRU', 'policy-agnostic: FIFO', 'policy-agnostic: LRU');
set(h,'FontSize',legendsize);
% error_fifo_mean = mean(Probes_fifo,2);
% error_lru_mean = mean(Probes_lru,2);
% error_fifo_var = var(Probes_fifo,0,2);
% error_lru_var = var(Probes_lru,0,2);
% figure; % probing cost
% plot(N, reshape(cost(:,1,1),size(N)), 'rs-',...
%     N, reshape(cost(:,1,2),size(N)), 'bo-', 'LineWidth', 1.5);
% xlabel('#repetitions n')
% ylabel('#probes')
% legend('FIFO', 'LRU')