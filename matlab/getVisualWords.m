function [wordMap] = getVisualWords(img, filterBank, dictionary)
% Compute visual words mapping for the given image using the dictionary of visual words.

% Inputs:
% 	img: Input RGB image of dimension (h, w, 3)
% 	filterBank: a cell array of N filters
% Output:
%   wordMap: WordMap matrix of same size as the input image (h, w)

    % TODO Implement your code here
%%
filterBank  = createFilterBank();
% dict = load('dictionary.mat');
% dictionary = dict.dictionary;
% img = imread('..\data\ice2.jpg');
size_img = size(img);
%imgDim = size(img);
%% Determining to which cluster each pixel belongs to.
filterResponses=extractFilterResponses(img,filterBank);
for h = 1:size_img(1)
    for w = 1:size_img(2)
        fR=reshape(filterResponses(h,w,:),1,60);
        distance = pdist2(fR,dictionary','euclidean');
         [D Index] = min(distance);
        % disp(min(distance));
        indices(h,w)= Index;
    end
end
        wordMap=indices;
         %imagesc(wordMap);
end

