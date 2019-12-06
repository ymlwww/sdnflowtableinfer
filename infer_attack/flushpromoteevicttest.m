function [ishit, cache, time_end] = flushpromoteevicttest(C_est,bgtraffic, time_begin, lm_a, cache, dI, policy)
% 'flush-promote-evict-test': return ishit (indicator of whether the last
% probe results in a hit
duration = bgtraffic(end,1);
testtrace=zeros(C_est+3,2);
testtrace(:,1) = poisson_trace(lm_a, C_est+3);
time_begin = mod(time_begin, duration);
testtrace(:,1) = testtrace(:,1) + time_begin; 
testtrace(C_est+3,1) = testtrace(C_est+3,1) + max(0, dI - (testtrace(C_est+3,1)-testtrace(C_est+2,1))+10^(-10)); % wait till evict is applied before testing
% if testtrace(end,1) > 2*duration
%     error('trace of background traffic is not long enough');
% end
shift = max([max(bgtraffic(:,2)) max(cache.C) max(cache.pend(:,2))]);
for i=1:C_est
    testtrace(i,2) = i + shift;
end
testtrace(C_est+1,2) = 1 + shift; 
testtrace(C_est+2,2) = C_est+1 + shift;
testtrace(C_est+3,2) = 2 + shift;
bgtraffic1 = bgtraffic; bgtraffic1(:,1) = bgtraffic(:,1) + duration;
bgtraffic = cat(1,bgtraffic,bgtraffic1); % extend bgtraffic by 2 (for wrap around)
bgtraffic = bgtraffic(bgtraffic(:,1)>=testtrace(1,1) & bgtraffic(:,1)<=testtrace(end,1) ,:); % only consider the background traffic during probing
mergetrace=cat(1,bgtraffic,testtrace(1:C_est+2,:));
mergetrace = sortrows(mergetrace,1);
Nreq = length(mergetrace(:,1));
for i=1:Nreq
    [cache, ~] = CacheAdd_withdelay(cache,mergetrace(i,1),mergetrace(i,2),dI,policy);
end
% test if testtrace(C+3,:) is a hit:
[cache, ishit] = CacheAdd_withdelay(cache,testtrace(C_est+3,1),testtrace(C_est+3,2),dI,policy);
time_end = testtrace(C_est+3,1);
% #probes: C_est+3 

end

