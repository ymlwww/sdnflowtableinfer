function [ T ] = characteristic_time_estimation( trace, tau, b, delta, Cinit, dI, policy )
% estimate the "characteristic time" as the minimum amount of idle time
% for a newly inserted content to be evicted from the cache (TTL)
% trace: N*2 matrix, trace(i,1): arrival time of i-th packet in background
% traffic; trace(i,2): content requested by i-th packet
% tau: initial guess of characteristic time
% b: #trials at each value of tau
% delta: stop when h-l<=delta
% Cinit: initial cache state (Cinit(i): index of content at position i,
% where i=1 is head, i=length(Cinit) is tail that will be evicted next.
% assume: content types are 0,1,...,N (unknown)
% dI: delay in inserting a new content after a miss
nc = -1; % index of new content (used for probing)
duration = trace(end,1); 
l = 0; h = inf;
cache.C = Cinit;
cache.pend = zeros(0,2); 
t = 0; % time in trace
% send first probe
[cache,~] = CacheAdd_withdelay( cache, t, nc, dI, policy );
% switch policy
%     case 'FIFO'
%         Cache = FifoAdd(Cinit, nc);
%     case 'LRU'
%         Cache = LruAdd(Cinit, nc);
%     otherwise
%         error(['unknown policy: ' policy]);
% end
while h-l > delta
%     disp(['l = ' num2str(l) ', h = ' num2str(h) ', tau = ' num2str(tau) ])
    hits = 0;    
    for i=1:b
        r1 = find(trace(:,1)>= t, 1,'first');
        if trace(end,1) < t+tau % wrap-around if at the end of trace]
            disp(['characteristic time estimation: wrap around trace'])
            trace(1:r1-1,1) = trace(1:r1-1,1) + duration; 
            trace = cat(1, trace(r1:end,:), trace(1:r1-1,:));
            r1 = 1;
        end
%         if trace(end,1)>=t+tau
            r2 = find(trace(:,1)<t+tau, 1, 'last'); % trace(r1:r2,:) arrive in [t, t+tau)
%             interval = r1:r2;
%         else % wrap-around if at the end of trace
%             r2 = find(trace(:,1)< t+tau-trace(end,1), 1, 'last');
%             interval = [1:r2 r1:length(trace(:,1))];
%         end
        % feed background traffic
        for r = r1:r2 %interval 
            [cache,~] = CacheAdd_withdelay( cache, trace(r,1), trace(r,2), dI, policy );
        end
        [cache, ishit] = CacheAdd_withdelay( cache, t+tau, nc, dI, policy );
        hits = hits + ishit;
%         switch policy
%             case 'FIFO'
%                 for r = interval 
%                     Cache = FifoAdd(Cache, trace(r,2));
%                 end
%                 if any(Cache == nc)
%                     hits = hits+1;
%                 end
%                 Cache = FifoAdd(Cache, nc);                
%             case 'LRU'
%                 for r = interval
%                     Cache = LruAdd(Cache, trace(r,2));
%                 end
%                 if any(Cache == nc)
%                     hits = hits+1;
%                 end
%                 Cache = LruAdd(Cache, nc);  
%             otherwise
%                 error(['unknown policy: ' policy]);
%         end   
        t = t + tau;
%         if trace(end,1)>=t+tau
%             t = t + tau;
%         else 
%             t = t + tau - trace(end,1);
%             disp(['characteristic time estimation: wrap around trace'])
%         end
    end
%     disp(['hit ratio ' num2str(hits/b)])
    
    if hits < b/2
        h = tau; 
        tau = (l+tau)/2;
    else
        l = tau;
        if h < inf
            tau = (tau+h)/2;
        else
            tau = 2*tau;
        end
    end    
end
T = (l+h)/2;

end

