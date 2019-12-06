function [hits, probes, cache, time_end] = forwardbackwardprobing(c,bgtraffic, time_begin, F, lm_a, cache, dI, policy)
% c: send a sequence of c distinct probes and then reverse the sequence
% bgtraffic: trace of background traffic, assumed to start from time 0
% F: #contents in background traffic
% lm_a: probing rate
% cache: initial cache state
% policy: 'FIFO' or 'LRU'
% time: initial time (end of last experiment)
% return: #hits of probes (hits), #probes sent (probes), final cache state (cache), ending time
% (time_end)
% cache = [0:999]'; % empty cache

duration = bgtraffic(end,1); % bgtraffic covers time [0, duration]
testtrace = zeros(2*c,2);
testtrace(:,1) = poisson_trace(lm_a, 2*c); % probing experiment starts from time 0
time_begin = mod(time_begin, duration);
testtrace(:,1) = testtrace(:,1) + time_begin; 
testtrace(c+1:2*c,1) = testtrace(c+1:2*c,1) + max(0, dI-(testtrace(c+1,1)-testtrace(c,1))+10^(-10)); % wait till all c contents are inserted before testing
if testtrace(end,1) > 2*duration
    error('trace of background traffic is not long enough');
end
shift = max([F max(cache.C) max(cache.pend(:,2))]); % make sure the probes are for new contents
for i = 1:c
    testtrace(i,2) = i+shift;
    testtrace(2*c+1-i,2) = i+shift; 
end
bgtraffic1 = bgtraffic; bgtraffic1(:,1) = bgtraffic(:,1) + duration;
bgtraffic = cat(1,bgtraffic,bgtraffic1); % extend bgtraffic by 2 (for wrap around)
bgtraffic = bgtraffic(bgtraffic(:,1)>=testtrace(1,1) & bgtraffic(:,1)<=testtrace(end,1) ,:); % only consider the background traffic during probing
mergetrace=cat(1,bgtraffic,testtrace);
mergetrace = sortrows(mergetrace,1);
Nreq = length(mergetrace(:,1));
hits = 0;
for i=1:Nreq
    id_curr = mergetrace(i,2);
    time_curr = mergetrace(i,1);
    [cache, ishit] = CacheAdd_withdelay(cache,time_curr,id_curr,dI,policy);
    if time_curr>=testtrace(c+1,1) && id_curr > shift % if it is a probe in the backward probing phase
        hits = hits + ishit;
        if ishit == 0
            break;
        end
    end
end
time_end = time_curr;
if hits < c
    probes = c + hits + 1;
else% hits == c
    probes = c + hits;
end

end