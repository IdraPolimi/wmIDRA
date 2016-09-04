function x = Normalize( x, x_min, x_max )
    x = (x - x_min) ./ (x_max - x_min);
end

