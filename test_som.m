function test_som(r_imgs)
    
    
    in = r_imgs;
    
    som_dim_1 = 50;
    som_dim_2 = 80;
    in_dim = size(in,2);
    
    mm = max(max(in));
    som = mm.*rand(som_dim_1,som_dim_2, in_dim) - mm / 2;
    
    
    figure(1);
    p = mesh(zeros(som_dim_1, som_dim_2));
    figure(2);
    f1 = mesh(som(:,:,1));
    figure(3);
    f2 = mesh(som(:,:,2));
        
        
    lr = 0.5*ones(som_dim_1,som_dim_2);
    a = 0.8;
    b = 0.6;
    as = 0.33;
    bs = 0.33;
    sf = 250;
    sg = 250;
    s = 2*rand(1,in_dim);
    E = 0.5 * rand(1, in_dim) + 0.1;
    E2 = 0.5 * rand(1, in_dim) + 0.1;
    R = 50*ones(som_dim_1,som_dim_2);
    
    NH = zeros(som_dim_1,som_dim_2,som_dim_1,som_dim_2);
    
    indeces = zeros(som_dim_1,som_dim_2, 2);
    
    for ii = 1:som_dim_1
        for jj = 1:som_dim_2
            indeces(ii,jj,:) = [ii, jj];
            if ii < som_dim_1
                NH(ii,jj,ii + 1,jj) = 1;
            end
            if ii > 1
                NH(ii,jj,ii - 1,jj) = 1;
            end
            if jj < som_dim_2
                NH(ii,jj,ii,jj + 1) = 1;
            end
            if jj > 1
                NH(ii,jj,ii,jj - 1) = 1;
            end
        end
    end
    
    
%     in(1001:2000, :) = repmat([500 , 0], 1000, 1)  + 5*rand(1000,2);
%     in(2001:3000, :) = repmat([0 , 600], 1000, 1)  + 5*rand(1000,2);
%     in(2001:3000, :) = repmat([100, 200], 1000, 1)  + 5*rand(1000,2);
%     in(3001:4000, :) = repmat([-50,150], 1000, 1)  + 10*rand(1000,2);
%     in(4001:5000, :) = repmat([-300,-100], 1000, 1)  + 10*rand(1000,2);
   
%     
%     in(1:5000,:) = in(randperm(5000),:);
%     in(1:5000,:) = in(randperm(5000),:);
    in = in(randperm(length(in)),:);
    nn = 1;
    while 1
        nn
        x = in(nn,:);
        nn = nn + 1;
        
        if nn > 500
            
        end
        
        dists_x = get_som_dists(som, x, s);
        
        %-------------- find BMU
        
        [m1, m2] = find(dists_x == min(min(dists_x)));
        
        m1 = m1(1); m2 = m2(1);
        
        %------------- update neighbourhood
        
        nh = reshape(NH(m1, m2, :, :), som_dim_1, som_dim_2);
        dists_w = get_som_dists(som, som(m1,m2,:), s);
        
        mean_neighbor_dist = sum(sum(dists_w .* nh) ./ (sum(sum(nh))*sg));
        
        R(m1,m2) = R(m1,m2) + b.*(func_g(som, mean_neighbor_dist) - R(m1,m2));
        
        indd = cat(3, repmat(m1, som_dim_1, som_dim_2), repmat(m2, som_dim_1, som_dim_2));

        
        dists = indeces - indd;
        dists = sqrt(sum(dists.^2,3));
        
        NH(m1, m2,:,:) = dists <= R(m1,m2);
        
        %------------- update learning rate
        
        
        lr = lr + nh .* a .* ( func_f(dists_x / sf) - lr );
        
        
        inn = reshape(x,1,1,in_dim);
        inn = repmat(inn, som_dim_1, som_dim_2, 1);
        nh(m1,m2) = 1;
        llr = repmat(lr,1,1,in_dim) .* repmat(nh,1,1,in_dim);
        
        som = som + llr .* (inn - som);
        
        
        E2 = E2 + as .* (x.^2 - E2);
        E = E + bs .* (x - E);
        
        s = sqrt(max(0, E2 - E.^2));
        
        
        if mod(nn,20) == 0
            plot_mean_dist(som,p);
        end
        
        f1.ZData = som(:,:,1);
        f2.ZData = som(:,:,2);
        
        pause(0.01);
    end
    
end

function plot_mean_dist(som, p)
    som_dim = size(som);
    data = zeros(som_dim(1), som_dim(2));
    tmp = zeros(som_dim(1), som_dim(2),2);
    
    
    
    %----- 1
    kern = [0, -1, 1];
    for ii = 1:som_dim(3)

        cc = conv2(som(:,:,ii),kern);
        tmp(:,:,ii) = cc(:,3:end);

        
    end
    tmp = tmp .^2;
    data = data + sqrt(sum(tmp,3));
    
    %----- 2
    kern = [1, -1, 0];
    for ii = 1:som_dim(3)
        cc = conv2(som(:,:,ii),kern);
        tmp(:,:,ii) = cc(:,1:end-2);
    end
    tmp = tmp .^2;
    data = data + sqrt(sum(tmp,3));
    
    %----- 3
    kern = [0; -1; 1];
    for ii = 1:som_dim(3)
        cc = conv2(som(:,:,ii),kern);
        tmp(:,:,ii) = cc(3:end,:);
    end
    tmp = tmp .^2;
    data = data + sqrt(sum(tmp,3));
    
    %----- 4
    kern = [1; -1; 0];
    for ii = 1:som_dim(3)
        cc = conv2(som(:,:,ii),kern);
        tmp(:,:,ii) = cc(1:end-2,:);
    end
    tmp = tmp .^2;
    data = data + sqrt(sum(tmp,3));
    
    %---- normalize
    data = data ./4;
    
    data(1,:) = 0;
    data(end,:) = 0;
    data(:,1) = 0;
    data(:,end) = 0;
    
    p.ZData = data;
    refreshdata(p);
end

function mm = get_mean_neighbour_dist(som,ii,jj)
    som_dim = size(som);

    aa = ii - 1;
    bb = ii + 1;
    cc = jj - 1;
    dd = jj + 1;
    
    if aa == 0
        aa = som_dim(1);
    else
        if aa == som_dim(1) + 1
            aa = 1;
        end
    end
    
    if bb == 0
        bb = som_dim(1);
    else
        if bb == som_dim(1) + 1
            bb = 1;
        end
    end
    
    if cc == 0
        cc = som_dim(2);
    else
        if cc == som_dim(2) + 1
            cc = 1;
        end
    end
    
    if dd == 0
        dd = som_dim(2);
    else
        if dd == som_dim(2) + 1
            dd = 1;
        end
    end
    
    som_ii_jj = reshape(som(ii,jj,:), 1, som_dim(3));
    som_aa_jj = reshape(som(aa,jj,:), 1, som_dim(3));
    som_bb_jj = reshape(som(bb,jj,:), 1, som_dim(3));
    som_ii_cc = reshape(som(ii,cc,:), 1, som_dim(3));
    som_ii_dd = reshape(som(ii,dd,:), 1, som_dim(3));
    
    dist(1) = pdist(cat(1, som_ii_jj, som_aa_jj));
    dist(2) = pdist(cat(1, som_ii_jj, som_bb_jj));
    dist(3) = pdist(cat(1, som_ii_jj, som_ii_cc));
    dist(4) = pdist(cat(1, som_ii_jj, som_ii_dd));
    mm = mean(dist);
    
end


function dists = get_som_dists(som, vector, s)
    som_dim = size(som);
    
    ssom = reshape(som, som_dim(1) * som_dim(2), som_dim(3));
    
    vector = reshape(vector, 1, som_dim(3));
    vector = repmat(vector, som_dim(1) * som_dim(2), 1);
    
    ss = repmat(s, som_dim(1) * som_dim(2), 1);
    
    diff_xw = ssom - vector;
    
    dists = sqrt( sum( (diff_xw ./ ss).^2, 2 ) );
    dists = reshape(dists,som_dim(1), som_dim(2));
end

function y = func_f(x)
    s = size(x);
    one = ones(s);
    
    y = one - one ./ (one + x);
end

function y = func_g(som, x)
    s = size(x);
    one = ones(s);
    
    som_dim = size(som);
    M = sqrt(som_dim(1).^2 + som_dim(2).^2);
    y = (M*1.4142*one - one ).*(one - one ./ (one + x));
end
