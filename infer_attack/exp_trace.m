function trace = exp_trace(F, alpha, lm, time_end)
% F: total #contents (i.e., #flows by legitimate users)
% alpha: skewness parameter of Zipf distribution of flow sizes
% lm: total rate
% time_end: duration (in seconds)
lambda = (1./(1:F).^alpha);
%disp(lambda);
lambda = lambda./sum(lambda);
%disp(lambda);
cs=cumsum(lambda); % upper bin boundaries: if rand in [0,cs(1)), it is for content 0, if in [cs(1), cs(2)), it is for content 1,...
% xxx=1:N;
% lambda(1)
% plot(xxx,lambda);

T=exprnd(1/lm,ceil(lm*time_end + sqrt(lm*time_end)),1); % Ting: fixed bug; include all arrivals from 0 to time_end
T=cumsum(T);
while T(end) < time_end
    T = [T; T(end) + exprnd(1/lm,1,1)];
end
if T(end)> time_end 
    T=T(1:find(T>time_end,1)-1);
end
trace = zeros(length(T),2);
trace(:,1)=T;
randarr=rand(length(T),1);
for i=1:length(T)
    trace(i,2)= find(cs>randarr(i),1)-1; % content index, in 0,...,F-1
end
end
