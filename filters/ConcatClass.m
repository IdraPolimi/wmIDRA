function x = ConcatClass( x, ind, N)

    len_in = size(x, 1);
    meta = zeros(len_in, N);
    meta(:, ind) = 1;
    x = cat(2, x, meta);
    
end

