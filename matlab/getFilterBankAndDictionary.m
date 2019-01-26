function [filterBank, dictionary] = getFilterBankAndDictionary(imPaths)
% Creates the filterBank and dictionary of visual words by clustering using kmeans.

% Inputs:
%   imPaths: Cell array of strings containing the full path to an image (or relative path wrt the working directory.
% Outputs:
%   filterBank: N filters created using createFilterBank()
%   dictionary: a dictionary of visual words from the filter responses using k-means.
% TODO Implement your code here

filterBank  = createFilterBank();
alpha = 100;
%% Generating filterResponses first (better approach).
% Method : Applying Filter for each image after reading, then selecting
% random pixels later  
for i = 1:length(imPaths)
        I = imread(imPaths{i,1});
        H_I = size(I,1); W_I = size(I,2); 
        randomMatrix = randperm(H_I*W_I,alpha);
        ch_no = size(I,3);
        L = 1+alpha*(i-1);
        U = alpha*i;
        % If Image is grayscale use repmat() to replicate channels
        if ch_no == 1
            I = repmat(I,[1 1 3]);
        end
        filterResponses = extractFilterResponses(I, filterBank);
        size_fR1 = size(filterResponses,1);
        size_fR2 = size(filterResponses,2);
        size_fR3 = size(filterResponses,3);
        filter2d = reshape(filterResponses, size_fR1*size_fR2, size_fR3);
        alphaResponse(L:U,:) = filter2d(randomMatrix,:);
end 
        
k=200;
[~, dictionary] = kmeans(alphaResponse, k, 'EmptyAction', 'drop');
dictionary=dictionary';
% Method : end


end


