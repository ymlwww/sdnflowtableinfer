function totaltrace = attack_trace(lambda, time_end, F)
% generate attack traffic: 
% lambda: 1*n array, lambda(i) is the rate of
% attack flow i (Poisson)
% time_end: duration of trace (in second)
% F: first flow will be for content F+i (background traffic requests
% contents 0,...,F-1)
totaltrace = [];
for i=1:length(lambda)
    lm = lambda(i);
    if lm == 0
        continue;
    else
        T=exprnd(1/lm,ceil(lm*time_end + sqrt(lm*time_end)),1);
        T=cumsum(T);
        while T(end) < time_end
            T = [T; T(end) + exprnd(1/lm,1,1)];
        end
        if T(end)> time_end
            T=T(1:find(T>time_end,1)-1);
        end
        trace = zeros(length(T),2);
        trace(:,1)=T;
        trace(:,2)= F+i;
        totaltrace=cat(1, totaltrace, trace);
    end
end
