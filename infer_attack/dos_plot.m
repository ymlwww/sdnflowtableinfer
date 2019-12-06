load('data/dos_equalrate_delay3.mat');
h=figure;
plot(lm_a, H_fifo_s, 'rs',...
    lm_a1, H_fifo_t, 'r-',...
    lm_a, H_lru_s, 'bo',...
    lm_a1, H_lru_t, 'b--', 'LineWidth', 1.5);
legend('FIFO with correct paramters', 'FIFO with correct paramters', 'LRU with correct paramters', 'LRU with estimated paramters')
xlabel('attack traffic rate (per flow)')
ylabel('hit ratio for legitimate users')