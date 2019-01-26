function [filterResponses] = extractFilterResponses(img, filterBank)
% Extract filter responses for the given image.
% Inputs: 
%   img:                a 3-channel RGB image with width W and height H
%   filterBank:         a cell array of N filters
% Outputs:
%   filterResponses:    a W x H x N*3 matrix of filter responses
% TODO Implement your code here

%% To Covert RGB2Lab and Check if image is grayScale or not.
%img = imread('..\data\ice.jpg'); 
imd = im2double(img);
%filterBank=createFilterBank();
%[row, col, ch] = size(I);
ch_no = size(imd,3);
if ch_no == 1
      imd = repmat(imd,[1 1 3]);
end
I = RGB2Lab(imd);
[L, a, b] = RGB2Lab(I(:,:,1), I(:,:,2), I(:,:,3));
%% This is to generate a filterResponse matrix of size W x H x F*3 of Fresponses
filterResponses=zeros(size(I,1),size(I,2),size(I,3));
for i=1:size(filterBank)
    j=(i-1)*3;
    filter=imfilter(I, filterBank{i});
    filterResponses(:,:,j+1) = filter(:,:,1); % L Channel
    filterResponses(:,:,j+2) = filter(:,:,2); % A Channel
    filterResponses(:,:,j+3) = filter(:,:,3); % B Channel
end

%% For generating montage uncomment the below part and comment the above section.
% filters=imfilter(I,filterBank{1});
% for i=2:size(filterBank)
%         filter_response = imfilter(I, filterBank{i});
%         disp(size(filter_response));
%         filters=cat(4,filters,filter_response);
% end
% 
% montage(filters,'size',[4 5]);
end
