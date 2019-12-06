function [ cache, ishit ] = CacheAdd_withdelay( cache, t, id, dI, policy )
% insert content 'id' into a cache with state 'cache' and replacement
% policy 'policy' at time 't'
% dI: insertion delay (to fetch a new content/install a new rule)
% cache.C: cache content (ordered)
% cache.pend: time/id of pending insertions (due to dI)
% return: updated cache state 'cache', indicator of hit 'ishit'

% apply pending new contents that will be installed before current time
if any(cache.pend(:,1)<t)
    cache.C = CacheAdd(cache.C, cache.pend(cache.pend(:,1)<t, 2), policy);
end
cache.pend = cache.pend(cache.pend(:,1)>=t,:);
% feed the new request to cache
stat1 = 0;
switch policy
    case 'FIFO'
        stat1 = any(cache.C == id);
    case 'LRU'
        stat1 = any(cache.C == id);
    case 'qLRU'
        stat1 = any(cache.C == id);
    case 'Random'
        stat1 = any(cache.C == id);
    case 'Lru2'
        stat1 = any(cache.C(:,2) == id);
    case 'Random2'
        stat1 = any(cache.C(:,2) == id);
    case 'ARC'
        stat1 =   any(cache.C(:,2) == id) ||  any(cache.C(:,3) == id);
    otherwise
        error(['unknown policy delay: ' policy]);
end
if stat1 % cache hit
    ishit = 1;
    cache.C = CacheAdd(cache.C, id, policy);
else % cache miss
    ishit = 0;
    if ~any(cache.pend(:,2)==id) % aggregate pending requests
        cache.pend = [cache.pend; [t+dI id]];
    end
end


end

