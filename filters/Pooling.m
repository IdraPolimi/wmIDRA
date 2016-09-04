function [ imgs ] = Pooling( x, patch_size)

s = size(x);

x = reshape(x, s(1), patch_size, patch_size);
imgs = zeros(s(1), floor(patch_size/2).^2);

    for ii = 1:s(1)
        curr = reshape(x(ii, :, :), patch_size, patch_size);
        for jj = 1:2:patch_size

            curr(jj, :) = max(curr(jj:jj+1, :), [], 1);
        end
        curr(1:2:patch_size, :) = [];
        for jj = 1:2:patch_size

            curr(:, jj) = max(curr(:, jj:jj+1), [], 2);
        end
        curr(:, 1:2:patch_size) = [];


        imgs(ii, :) = reshape(curr, 1, floor(patch_size/2).^2);
    end

end

