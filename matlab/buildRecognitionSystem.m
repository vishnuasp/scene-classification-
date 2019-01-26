function buildRecognitionSystem()
% Creates vision.mat. Generates training features for all of the training images.

	load('dictionary.mat');
	load('../data/traintest.mat');

	% TODO create train_features
%% To calculate K*... x T matrix using getImageFeaturesSPM


	interval= 1;
    layerNum = 3;
    dict = load('dictionary.mat');
    dictionary = dict.dictionary;
    dictionarySize = size(dictionary,2);
	train_names = train_imagenames(1:interval:end);
    for i=1:length(train_names)
        load(['..\data\',strrep(train_names{i},'.jpg','.mat')]);
        h = getImageFeaturesSPM(layerNum, wordMap, dictionarySize);
        train_features(:,i) = h;
    end
%%

	save('vision.mat', 'filterBank', 'dictionary', 'train_features', 'train_labels');

end