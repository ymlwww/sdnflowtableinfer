function [ h_fifo, h_lru,h_qlru, h_random, h_lru2, h_random2, h_arc] = hit_ratio_with_attack_sim(cachesize, lm_background, alpha, F, lm_attack,time_end, dI, inputFileName)
% 

% simulate the actual hit ratio for background traffic for a FIFO or LRU
% cache, under an attack of length(lm_attack) flows of rates in lm_attack
% (1*n array for n attack flows)
% alpha=0.9;
% F = 5000;
% dI = 0;
testtrace = attack_trace(lm_attack,time_end, F);
trace=cell2mat(struct2cell(load(inputFileName)));
bgtraffic = trace.';
% bgtraffic = exp_trace(F, alpha, lm_background, time_end);
mergetrace=cat(1,bgtraffic,testtrace);
mergetrace = sortrows(mergetrace,1);
C_fifo.C = -ones(cachesize,1); % cache state (initially empty)
C_lru.C = C_fifo.C; 
C_qlru.C = C_fifo.C; 
C_random.C = C_fifo.C; 
C_lru2.C = -ones(cachesize,2); 
C_random2.C = C_lru2.C; 
c1 = int16.empty(0); 
c2 = int16.empty(0); 
c3 = int16.empty(0); 
c4 = int16.empty(0); 
y_fifo = 0; % #hits for background traffic under FIFO
y_lru = 0; % #hits for background traffic under LRU
y_qlru = 0; 
y_random = 0; 
y_lru2 = 0; 
y_random2 = 0; 
y_arc = 0; 
C_fifo.pend = zeros(0,2); % store (time, content_id) for new contents entering the cache (due to new rule installation delay)
C_lru.pend = zeros(0,2);
C_qlru.pend = zeros(0,2);
C_random.pend = zeros(0,2);
C_lru2.pend = zeros(0,2);
C_random2.pend = zeros(0,2);
pend = zeros(0,2);
Nreq = length(mergetrace(:,1)); 
for i=1:Nreq
    t = mergetrace(i,1); % current time
    id_curr = mergetrace(i,2);
    [C_fifo, ishit] = CacheAdd_withdelay( C_fifo, t, id_curr, dI, 'FIFO');
    if id_curr < F % a request from legitimate users
        y_fifo = y_fifo + ishit;
    end
    [C_lru, ishit] = CacheAdd_withdelay( C_lru, t, id_curr, dI, 'LRU');
    if id_curr < F % a request from legitimate users
        y_lru = y_lru + ishit;
    end
    [C_qlru, ishit] = CacheAdd_withdelay( C_qlru, t, id_curr, dI, 'qLRU');
    if id_curr < F % a request from legitimate users
        y_qlru = y_qlru + ishit;
    end
    [C_random, ishit] = CacheAdd_withdelay( C_random, t, id_curr, dI, 'Random');
    if id_curr < F % a request from legitimate users
        y_random = y_random + ishit;
    end
    [C_lru2, ishit] = CacheAdd_withdelay( C_lru2, t, id_curr, dI, 'Lru2');
    if id_curr < F % a request from legitimate users
        y_lru2 = y_lru2 + ishit;
    end
    [C_random2, ishit] = CacheAdd_withdelay( C_random2, t, id_curr, dI, 'Random2');
    if id_curr < F % a request from legitimate users
        y_random2 = y_random2 + ishit;
    end
    [c1, c2, c3 ,c4 , pend ,ishit] = CacheAdd_withdelay_arc(pend, c1, c2, c3 ,c4, t, id_curr, dI);
    if id_curr < F % a request from legitimate users
        y_arc = y_arc + ishit;
    end
end
h_fifo = y_fifo / length(bgtraffic(:,1));
h_lru =  y_lru / length(bgtraffic(:,1));
h_qlru =  y_qlru / length(bgtraffic(:,1));
h_random =  y_random / length(bgtraffic(:,1));
h_lru2 =  y_lru2 / length(bgtraffic(:,1));
h_random2 =  y_random2 / length(bgtraffic(:,1));
h_arc =  y_arc / length(bgtraffic(:,1));
end