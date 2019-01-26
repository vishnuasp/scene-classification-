function [h] = getImageFeaturesSPM(layerNum, wordMap, dictionarySize)
% Compute histogram of visual words using SPM method
% Inputs:
%   layerNum: Number of layers (L+1)
%   wordMap: WordMap matrix of size (h, w)
%   dictionarySize: the number of visual words, dictionary size
% Output:
%   h: histogram of visual words of size {dictionarySize * (4^layerNum - 1)/3} (l1-normalized, ie. sum(h(:)) == 1)

    % TODO Implement your code here
    
% layerNum = 3;
% finestL = 2^(layerNum - 1);
% dict = load('dictionary.mat');
% dictionary = dict.dictionary;
% filterBank = createFilterBank();
% dictionarySize = size(dictionary,2);
% h = zeros((dictionarySize * (4^layerNum - 1)/3),1);
% img = imread('..\data\ice2.jpg');
% wordMap = getVisualWords(img, filterBank, dictionary);
%% Evaluating histogram of visual words.
% Level 2
size_WM = size(wordMap);
finestL = 2^(layerNum - 1);
rows = floor(size_WM(1)/finestL);
cols = floor(size_WM(2)/finestL);
partitionNo = 0;
for i = 1:finestL
    for j=1:finestL
        miniMap = wordMap((1+(i-1)*rows:i*rows),(1+(j-1)*cols:j*cols));
        mini_h = getImageFeatures(miniMap, dictionarySize);
        partitionNo = (i-1)*2^(layerNum-1) + j;
        h(1+(partitionNo-1)*dictionarySize:partitionNo*dictionarySize, 1) = mini_h;
     end
end
% Level 1
rows1=floor((2*size_WM(1))/finestL);
cols1=floor((2*size_WM(2))/finestL);
for i = 1: (finestL/2)
    for j=1: (finestL/2)
        miniMap = wordMap((1+(i-1)*rows1:i*rows1),(1+(j-1)*cols1:j*cols1));
        mini_h = getImageFeatures(miniMap, dictionarySize);
        partitionNo = partitionNo+1;
        h(1+(partitionNo-1)*dictionarySize:partitionNo*dictionarySize, 1) = mini_h;
     end
end
% Level 0
partitionNo = partitionNo+1;
mini_h = getImageFeatures(wordMap, dictionarySize);
h(1+(partitionNo-1)*dictionarySize:partitionNo*dictionarySize, 1) = mini_h;
h = h/ norm(h,1);

end