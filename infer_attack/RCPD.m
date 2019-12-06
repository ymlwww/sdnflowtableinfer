function [policy_index, probes] = RCPD(C_est,C,N,bgtraffic,lm_a,dI,policy)
% Robust Cache Policy Detection (size-aware cache policy inference):
probes = 0; 
cache.C = -ones(C,1);
cache.pend = zeros(0,2);
time_end = bgtraffic(1,1)-eps;

for i=1:N
    disp(['trying i = ' num2str(i) ', beginning at time ' num2str(time_end)])
    [ishit, cache, time_end] = flushpromoteevicttest(C_est,bgtraffic,time_end,lm_a,cache,dI,policy);
    probes = probes + C_est + 3;
    if ishit > 0
        policy_index=0; % means 'FIFO'
        return;
    end
end
policy_index=1; % means 'LRU'

end

