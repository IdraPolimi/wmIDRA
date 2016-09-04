function [icasig, A, W] = doica(sig, n)
%ica   compute the ICA on the signals sig
%
%   sig must be a matrix; each row of sig is one observed signal;
%   the row of icasig are the estimated IC;
%   A is the mixing matrix (sig = A*icasig);
%   W is the separating matrix (icasig = W*sig);

epsilon = 0.0001;
maxNumIterations = 10000;

[Dim, NumOfSampl] = size(sig)

sig = gpuArray(sig);

% Remove the mean and check the data
%mixedsig = zeros(size(sig));
mixedmean = mean(sig')';

%asd = mixedmean * ones(1,size(sig, 2));

mixedsig = sig - mixedmean * ones(1,size(sig, 2));

%Calculate the covariance matrix.
covarianceMatrix = cov(mixedsig');

[E, D] = eig(covarianceMatrix);     

%%%%%%%%%%remove negative eigenvalues and corresponding eigenvector%%%%%%%
% negativeEigenvalues = find(D<0);
[rowIndexes, columnIndexes] = find(D<0);
D(:,columnIndexes)=[];
D(rowIndexes,:)=[];
E(:,columnIndexes)=[];
%%%%% check wheter there are negatives eigenvalues
diag(D)<0
if any(diag(D)<0)
    error('stICA_Final:eig','Some eigenvalues are negative due to MATLAB approximation');
end

%%%%%%%%%%%%%%%%%%%%%%%% Whitening %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
whiteningMatrix = inv (sqrt (D)) * E'; 
dewhiteningMatrix = E * sqrt (D);

whitesig =  whiteningMatrix * mixedsig;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ICA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dim = size(whitesig, 1);
A = zeros(Dim, n);
B = orth (randn (Dim, n));
BOld = zeros(size(B));
for step = 1:maxNumIterations + 1,
    % Symmetric orthogonalization.
    B = B * real(inv(B' * B)^(1/2));
    % Test for termination condition.
    minAbsCos = min(abs(diag(B' * BOld)));
    if (1 - minAbsCos < epsilon)
        fprintf('Convergence ok\n');
        % Calculate the de-whitened vectors.
        A = dewhiteningMatrix * B;
        break;
    end
    BOld = B;
    B = (whitesig * (( whitesig' * B) .^ 3)) / NumOfSampl - 3 * B;
end

% Calculate ICA filters.
W = B' * whiteningMatrix;
icasig = W * mixedsig + (W * mixedmean) * ones(1, NumOfSampl);

icasig = gather(icasig);
A=gather(A);
W=gather(W);
end