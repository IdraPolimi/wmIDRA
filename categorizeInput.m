function [updatedCategories, ids, meand, covmat] = categorizeInput(data,k)
% categorizeInput: categorize the new input with respect to
% data
% size(data)
% numberOfCategories
[ids, updatedCategories, sumd, dist] = kmeans(data, k, 'EmptyAction','singleton', 'Distance', 'sqeuclidean');



count = hist(ids,1:k) + 1;


%-------------------
mm = min(dist,[],2);

dd = dist == repmat(mm,1,size(dist,2));

sumd = sum(sqrt(dist.*dd));

meand = (sumd ./ count)';

meand = max(meand, 10^-6);

obs_size = size(updatedCategories, 2);

covmat = zeros(obs_size, obs_size, k);

for ii = 1:k;
    category_data = data(ids==ii,:);
    centered_data = category_data - repmat(updatedCategories(ii,:), size(category_data,1), 1);
    covmat(:,:,ii) = cov(centered_data) ;
end

end


