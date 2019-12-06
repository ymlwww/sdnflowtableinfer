function trace = poisson_trace(lm, c)
% generate a realization of a Poisson process with rate lm and c points
trace = exprnd(1/lm,c,1); % Ting: fixed bug; include all arrivals from 0 to time_end
trace=cumsum(trace);
% Note: this is only the timestamp, not including the content 
end


% function trace = poisson_trace(lm, time_end)
% % generate a realization of a Poisson process in [0, time_end] with rate lm
% % lm: total rate
% % time_end: duration (in seconds)
% r = poissrnd(lm*time_end); % #packets
% trace = sort(rand(r,1)*time_end); % conditioned on a Poisson process has r points in [0, time_end], the r points are uniformly distributed in this interval
% % Note: this is only the timestamp, not including the content 
% end
