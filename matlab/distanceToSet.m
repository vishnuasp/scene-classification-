function histInter = distanceToSet(wordHist, histograms)
% Compute distance between a histogram of visual words with all training image histograms.
% Inputs:
% 	wordHist: visual word histogram - K * (4^(L+1) − 1 / 3) × 1 vector
% 	histograms: matrix containing T features from T training images - K * (4^(L+1) − 1 / 3) × T matrix
% Output:
% 	histInter: histogram intersection similarity between wordHist and each training sample as a 1 × T vector

	% TODO Implement your code here
    

	load('dictionary.mat');
	load('../data/traintest.mat');

	% TODO create train_features
%% Evaluating the histogram intersection similarities.

    length = size(histograms,2);
    for i=1:length
    distances(:,i)= bsxfun(@min,wordHist, histograms(:,i));
    end
    for i=1:size(distances,2)
        histInter(i,:) = sum(distances(:,i));
    end
	
end