% error_fifo=zeros(10,20);
% error_lru=zeros(10,20);
% for q =1:20
%     inputFileName = ['data/dos_total',int2str(q),'.mat'];
%     load(inputFileName);    
%     for i=1:10
%         error_fifo(i,q) = abs(H_fifo_s(i) - H_fifo_t(i))/H_fifo_t(i);
%         error_lru(i,q) = abs(H_lru_s(i) - H_lru_t(i))/H_lru_t(i);
%     end
% end
% error_fifo_mean = zeros(11,1);
% error_lru_mean = zeros(11,1);
% error_fifo_var = zeros(11,1);
% error_lru_var = zeros(11,1);
% for i = 1:10
%     error_fifo_mean(i) = mean(error_fifo(i,:));
%     error_lru_mean(i) = mean(error_lru(i,:));
%     error_fifo_var(i) = var(error_fifo(i,:));
%     error_lru_var(i) = var(error_lru(i,:));
% end
% x = 0:.1:1;
% figure;
% errorbar(x, error_fifo_mean*100, error_fifo_var*10000);
% hold on;
% errorbar(x, error_lru_mean*100, error_lru_var*10000);
% xlabel('attack traffic rate (per flow)');
% ylabel('relative error of hit ratio');
% title("relative error");
% legend("fifo error","lru error");
load('data/dos_total_new3.mat');
%%
% load('data/dos_equalrate.mat');
% load('data/dos_equalrate_delay.mat');
% load('data/dos_equalrate_delay_zoomin.mat');
h=figure;
plot(lm_a_sim, H_fifo_s, 'rs',...
    lm_a, H_fifo_t, 'r-',...
    lm_a_sim, H_lru_s, 'bo',...
    lm_a, H_lru_t, 'b--', 'LineWidth', 1.5);
legend('FIFO experimental', 'FIFO theoretical', 'LRU experimental', 'LRU theoretical')
xlabel('attack traffic rate (per flow)')
ylabel('hit ratio for legitimate users')