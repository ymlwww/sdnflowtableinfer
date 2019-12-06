function [C,probecount] = RCSE(n,c0,trueC,bgtraffic,lm_a,dI,policy)
% RCSE: given n repetitions per forwardbackwardprobing experiment, initial
% guess of cache size c0, return C (estimated cache size) and probecount
% (#probes)
C=0;
c=c0; % >= 2
probecount=0;
F = max(bgtraffic(:,2));
% initial cache state:
cache.C = -ones(trueC,1); % empty
% cache = [0:trueC-1]'; % top-C in cache
cache.pend = zeros(0,2); 

time_end = bgtraffic(1,1)-eps; 
while true
    disp(['trying c = ' num2str(c)])
    for i = 1:n
%         disp(['begin at time ' num2str(time_end)])
        [delta, probes, cache, time_end] =forwardbackwardprobing(c,bgtraffic, time_end, F, lm_a, cache, dI, policy);
%         [delta, probes] = forwardbackwardprobing_mingli(c,10,.9,5000); % sanity check with Mingli's bug-fixed version
        probecount = probecount + probes;
        C = max(C,delta);
        if delta == c
            break;
        end
    end
    if C<c
        break;
    else
        c=2*c;
    end
end

end
