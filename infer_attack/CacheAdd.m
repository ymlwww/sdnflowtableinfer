function [ cache ] = CacheAdd( cache, ids, policy )
% insert contents 'ids' into a cache with state 'cache' and replacement
% policy 'policy'
switch policy
    case 'FIFO'
        cache = FifoAdd(cache, ids);
    case 'LRU'
        cache = LruAdd(cache, ids);
    case 'qLRU'
        cache = qLruAdd(cache, ids);
    case 'Random'
        cache = RandomAdd(cache, ids);
    case 'Lru2'
        cache = Lru2Add(cache, ids);
    case 'Random2'
        cache = Random2Add(cache, ids);
    case 'ARC'
        cache = ArcAdd(cache, ids);
    otherwise
        error(['unknown policy: ' policy]);
end

end

