function [conf] = evaluateRecognitionSystem()
% Evaluates the recognition system for all test-images and returns the confusion matrix

	load('vision.mat');
	load('../data/traintest.mat');
%% Evaluating the recognition system based on the Confusion Matrix c,
	% TODO Implement your code here
    c = zeros(8,8);
    test_names = test_imagenames(1:1:end);
    for i=1:size(test_imagenames,1)
        lab = test_labels(i);
        guessedImage = guessImage(strcat('..\data\',test_names{i}));
        arr = mapping;
        %ind = strfind(arr,guessedImage);
        ind=find(ismember(arr,guessedImage));
        c(lab,ind) = c(lab,ind) + 1;
    end
    disp(c);
    accuracy =  (trace(c) / sum(c(:)))*100;
    disp(strcat(' Accuracy is : ',accuracy));
end