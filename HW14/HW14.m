dataDir = fullfile(tempdir,"rit18_data"); 
downloadHamlinBeachMSIData(dataDir);


load(fullfile(dataDir,"rit18_data.mat"));


train_data = switchChannelsToThirdPlane(train_data);
val_data   = switchChannelsToThirdPlane(val_data);
test_data  = switchChannelsToThirdPlane(test_data);
Image Acquisition, Computer Vision, Image Processing & Deep Lea

save("train_data.mat","train_data");
imwrite(train_labels,"train_labels.png");


figure
montage(...
    {histeq(train_data(:,:,[3 2 1])), ...
    histeq(val_data(:,:,[3 2 1])), ...
    histeq(test_data(:,:,[3 2 1]))}, ...
    BorderSize=10,BackgroundColor="white")
title("RGB Component of Training, Validation, and Test Image (Left to Right)")


figure
montage(...
    {histeq(train_data(:,:,4)),histeq(train_data(:,:,5)),histeq(train_data(:,:,6))}, ...
    BorderSize=10,BackgroundColor="white")
title("Training Image IR Channels 1, 2, and 3 (Left to Right)")


figure
montage(...
    {train_data(:,:,7),val_data(:,:,7),test_data(:,:,7)}, ...
    BorderSize=10,BackgroundColor="white")
title("Mask of Training, Validation, and Test Image (Left to Right)")


classNames = [ "RoadMarkings","Tree","Building","Vehicle","Person", ...
               "LifeguardChair","PicnicTable","BlackWoodPanel",...
               "WhiteWoodPanel","OrangeLandingPad","Buoy","Rocks",...
               "LowLevelVegetation","Grass_Lawn","Sand_Beach",...
               "Water_Lake","Water_Pond","Asphalt"]; 


cmap = jet(numel(classNames));
B = labeloverlay(histeq(train_data(:,:,4:6)),train_labels,Transparency=0.8,Colormap=cmap);

figure
imshow(B)
title("Training Labels")
N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar(TickLabels=cellstr(classNames),Ticks=ticks,TickLength=0,TickLabelInterpreter="none");
colormap(cmap)


downloadTrainedNetwork(trainedUnet_url,dataDir);
load(fullfile(dataDir,"multispectralUnet.mat"));


predictPatchSize = [1024 1024];
segmentedImage = segmentMultispectralImage(test_data,net,predictPatchSize);


segmentedImage = uint8(test_data(:,:,7)~=0) .* segmentedImage;

figure
imshow(segmentedImage,[])
title("Segmented Image")


segmentedImage = medfilt2(segmentedImage,[7,7]);
imshow(segmentedImage,[]);
title("Segmented Image with Noise Removed")


B = labeloverlay(histeq(test_data(:,:,[3 2 1])),segmentedImage,Transparency=0.8,Colormap=cmap);

figure
imshow(B)
title("Labeled Segmented Image")
colorbar(TickLabels=cellstr(classNames),Ticks=ticks,TickLength=0,TickLabelInterpreter="none");
colormap(cmap)


vegetationClassIds = uint8([2,13,14]);
vegetationPixels = ismember(segmentedImage(:),vegetationClassIds);
validPixels = (segmentedImage~=0);

numVegetationPixels = sum(vegetationPixels(:));
numValidPixels = sum(validPixels(:));


percentVegetationCover = (numVegetationPixels/numValidPixels)*100;
fprintf("The percentage of vegetation cover is %3.2f%%.",percentVegetationCover);


imds = imageDatastore("train_data.mat",FileExtensions=".mat",ReadFcn=@matRead6Channels);


pixelLabelIds = 1:18;
pxds = pixelLabelDatastore("train_labels.png",classNames,pixelLabelIds);


dsTrain = randomPatchExtractionDatastore(imds,pxds,[256,256],PatchesPerImage=16000);


inputBatch = preview(dsTrain);
disp(inputBatch)


inputTileSize = [256,256,6];
lgraph = createUnet(inputTileSize);
disp(lgraph.Layers)

initialLearningRate = 0.05;
maxEpochs = 150;
minibatchSize = 16;
l2reg = 0.0001;

options = trainingOptions("sgdm",...
    InitialLearnRate=initialLearningRate, ...
    Momentum=0.9,...
    L2Regularization=l2reg,...
    MaxEpochs=maxEpochs,...
    MiniBatchSize=minibatchSize,...
    LearnRateSchedule="piecewise",...    
    Shuffle="every-epoch",...
    GradientThresholdMethod="l2norm",...
    GradientThreshold=0.05, ...
    Plots="training-progress", ...
    VerboseFrequency=20);


doTraining = false; 
if doTraining
    net = trainNetwork(dsTrain,lgraph,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save(fullfile(dataDir,"multispectralUnet-"+modelDateTime+".mat"),"net");
end


segmentedImage = segmentMultispectralImage(test_data,net,predictPatchSize);


imwrite(segmentedImage,"results.png");
imwrite(val_labels,"gtruth.png");


pxdsResults = pixelLabelDatastore("results.png",classNames,pixelLabelIds);
pxdsTruth = pixelLabelDatastore("gtruth.png",classNames,pixelLabelIds);


ssm = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,Metrics="global-accuracy");
