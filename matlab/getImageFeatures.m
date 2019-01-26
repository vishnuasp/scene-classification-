function [h] = getImageFeatures(wordMap, dictionarySize)
% Compute histogram of visual words
% Inputs:
% 	wordMap: WordMap matrix of size (h, w)
% 	dictionarySize: the number of visual words, dictionary size
% Output:
%   h: vector of histogram of visual words of size dictionarySize (l1-normalized, ie. sum(h(:)) == 1)

	% TODO Implement your code here
    
%     filterBank = createFilterBank();
%     dict = load('dictionary.mat');
%     dictionary = dict.dictionary;
%     dictionarySize=size(dictionary,2);
%     img = imread('..\data\ice.jpg');
%     wordMap = getVisualWords(img, filterBank, dictionary);

     %h = imhist(wordMap);
     %h = histogram(wordMap(:),dictionarySize);
     %h_norm = h_norm';
%% Generating histograms using hist and normalizing them.
    h = hist(wordMap(:),dictionarySize);
    n = norm(h,1);
    h = h/n;
    %h = h/sum(h);
    h=h';
    %histogram(h, dictionarySize);
	assert(numel(h) == dictionarySize);
end