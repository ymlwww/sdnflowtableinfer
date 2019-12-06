function [C]=LruAdd(C,ids)
% this function return the sorted vector of item ids contained in a LRU cache.
% the forst element of the vector is the last accessed one
% i is the item id that would be inserted in the cache in case not already contained in 
% C ist the vector of the item ids contained in the cache before the adding of i
for i = 1:length(ids)
    id = ids(i);
    k=find(C==id);
    if numel(k>0)
        if k>1
            C=[id; C(1:k-1); C(k+1:end)];
        end
    else
%         r=C(end);
        C=[id; C(1:end-1)];
    end
end

end
    