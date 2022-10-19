imagename = 'test_images/';
inputSize = [224 224 3];
I_o = imread('test_images/1000.jpg');
I = imresize(I,inputSize(1:2));
[bboxes1,scores1] = detect(detector,I);

tempname = "temp";
T = table(tempname,bboxes1);
for i = 1:3
    i_id = append( num2str(i+999),'.jpg');
    ImageId{i} = i_id;
    imagename = append(filename, i_id);

    
    I = imread(imagename);
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);
    
    
    bbox{i} = bboxes;
    
    
end

% imdsTrain = imageDatastore(I);
% bldsTrain = boxLabelDatastore(bboxes1);
imdsTrain = imageDatastore(T{:,'tempname'});
bldsTrain = boxLabelDatastore(T(:,'bboxes1'));

trainingData = combine(imdsTrain,bldsTrain);
changesize = size(I_o);
preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,changesize));

% 
%     figure;
% for i = 1:300
%     filename = append(imagename, num2str(i+999),'.jpg');
%     I = imread(filename);
%     I = imresize(I,inputSize(1:2));
%     temp = bbox{i};
%     annotatedImage = insertShape(I,'Rectangle',temp);
%     
%     imshow(annotatedImage);
%     pause(0.5);
% end

% for i = 1:300
%     filename = append(imagename, num2str(i+999),'.jpg');
%     I = imread(filename);
%     I = imresize(I,inputSize(1:2));
% %     temp = bbox{i};
% %     annotatedImage = insertShape(I,'Rectangle',temp);
%     
%     imshow(I);
%     hold on;
%     plot([X_min{i} X_max{i}], [Y_min{i} Y_max{i}]);
%     pause(0.5);
% end











function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
data{2} = bboxresize(boxEstimate,scale);
end
function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));
I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end
% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);
% Apply same transform to boxes.
boxEstimate=round(A{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
[B{2},indices] = bboxwarp(boxEstimate,tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);
% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end