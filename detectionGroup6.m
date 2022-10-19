data = xlsread('train_boundingboxes.csv'); 

%%train data
imagename = 'images/';
imagefilename = cell(900,1);
birds = cell(900,1);
for i = 1 : 900
    imagename = append(imagename, num2str(i-1),'.jpg');
    imagefilename{i} = imagename;
    temp = data(i,:); %[xmin ymin xmax ymax]
    birdbox = [temp(1), temp(2), abs(temp(3)-temp(1)), abs(temp(4)-temp(2))]; %[xmin ymin xmax-xmin ymax-ymin]
    birds{i} = birdbox;
    
    imagename = 'images/';  
end
T = table(imagefilename,birds);
imdsTrain = imageDatastore(T{:,'imagefilename'});
bldsTrain = boxLabelDatastore(T(:,'birds'));

trainingData = combine(imdsTrain,bldsTrain);

%%validation data
imagename = 'images/';
imagefilename = cell(100,1);
birds = cell(100,1);
for i = 1 : 100
    imagename = append(imagename, num2str(i+899),'.jpg');
    imagefilename{i} = imagename;
    temp = data(i+900,:); %[xmin ymin xmax ymax]
    birdbox = [temp(1), temp(2), abs(temp(3)-temp(1)), abs(temp(4)-temp(2))]; %[xmin ymin xmax-xmin ymax-ymin]
    birds{i} = birdbox;
    
    imagename = 'images/';  
end
T = table(imagefilename,birds);
imdsValid = imageDatastore(T{:,'imagefilename'});
bldsValid = boxLabelDatastore(T(:,'birds'));

validationData = combine(imdsValid,bldsValid);


%% pre-process


inputSize = [224 224 3];
preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)


featureExtractionNetwork = resnet50;
featureLayer = 'activation_34_relu';
numClasses = 1;
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer); 

augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


%%Train Faster R-CNN
options = trainingOptions('sgdm',...
    'MaxEpochs',3,...
    'MiniBatchSize',10,...
    'InitialLearnRate',5e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData,...
    'ExecutionEnvironment', 'gpu', ...
    'LearnRateSchedule','piecewise', ...
    'Plots','training-progress');
[detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);


%%Test & get results with .csv file

filename = 'test_images/';

ImageId = cell(300,1);
X_min = cell(300,1);
Y_min = cell(300,1);
X_max = cell(300,1);
Y_max = cell(300,1);
for i = 1:300
    i_id = append( num2str(i+999),'.jpg');
    ImageId{i} = i_id;
    imagename = append(filename, i_id);

    
    I = imread(imagename);
    Original_size = size(I);
    scale = Original_size(1:2)./inputSize(1:2);
    
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);
    
    bboxes = bboxresize(bboxes,scale);
    
    
    if ~isempty(bboxes)
       X_min{i} = bboxes(1);
       Y_min{i} = bboxes(2);
       X_max{i} = bboxes(3)+bboxes(1);
       Y_max{i} = bboxes(4)+bboxes(2);
    else
       X_min{i} = 0;
       Y_min{i} = 0;
       X_max{i} = 0;
       Y_max{i} = 0;
    end
   
end
T = table(ImageId, X_min, Y_min, X_max, Y_max);
T.Properties.VariableNames = {'ImageId',' X_min', ' Y_min',' X_max',' Y_max'};
writetable(T,'result.csv');




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
