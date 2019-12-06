function [C] = FifoAdd(C,ids)

for i = 1:length(ids)
    id = ids(i);
    if any(C == id)
        
    else
        C=[id; C(1:end-1)];
    end
end

end