inputSize = [224 224 3];
filename = 'test_images/';

ImageId = cell(300,1);
% bbox = cell(300,1);
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
%        X_min{i} = bboxes(1);
%        Y_min{i} = bboxes(2);
%        X_max{i} = bboxes(3);
%        Y_max{i} = bboxes(4);
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
%     bbox{i} = bboxes;
   
end
T = table(ImageId, X_min, Y_min, X_max, Y_max);
T.Properties.VariableNames = {'ImageId',' X_min', ' Y_min',' X_max',' Y_max'};
writetable(T,'result.csv');
return;