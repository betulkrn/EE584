% % Calibration images
% clc;close all;clear all;
% % Take calibration images
% cam = webcam(2);
% preview(cam);
% for i = 1:9
%     pause(5);
%     img = snapshot(cam);
%     figure
%     image(img);
%     imwrite (img, ['q1\image' num2str(i) '.jpg']);
% end
% clear cam
%%
clc;clear;close all;

images = imageDatastore(fullfile("*"));
imageFileNames = images.Files;

numImages = 9;
files = cell(1, numImages);
for i = 1:numImages
    files{i} = fullfile(sprintf('image%d.jpg', i));
end

% Display one of the calibration images
magnification = 25;
I = imread(files{1});
figure; imshow(I, InitialMagnification = magnification);
title("One of the Calibration Images");


% Detect the checkerboard corners in the images.
[imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames);

% Generate the world coordinates of the checkerboard corners in the
% pattern-centric coordinate system, with the upper-left corner at (0,0).
squareSize = 29; % in millimeters
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera.
imageSize = [size(I, 1), size(I, 2)];
cameraParams = estimateCameraParameters(imagePoints, worldPoints, ...
                                     ImageSize = imageSize);

% Evaluate calibration accuracy.
figure; showReprojectionErrors(cameraParams);
title("Reprojection Errors");

%%
imOrig = imread(fullfile("image1.jpg"));
figure; imshow(imOrig, InitialMagnification = magnification);
title("Input Image");


%%

% Since the lens introduced little distortion, use 'full' output view to illustrate that
% the image was undistored. If we used the default 'same' option, it would be difficult
% to notice any difference when compared to the original image. Notice the small black borders.
[im, newOrigin] = undistortImage(imOrig, cameraParams, OutputView = "full");
figure; imshow(im, InitialMagnification = magnification);
title("Undistorted Image");


%%

% Convert the image to the HSV color space.
imHSV = rgb2hsv(im);

% Get the saturation channel.
saturation = imHSV(:, :, 2);

% Threshold the image
t = graythresh(saturation);
imId = (saturation < t);

figure; imshow(imId, InitialMagnification = magnification);
title("Segmented Ids");


%%

% Find connected components.
blobAnalysis = vision.BlobAnalysis(AreaOutputPort = true,...
    CentroidOutputPort = false,...
    BoundingBoxOutputPort = true,...
    MinimumBlobArea = 350, ExcludeBorderBlobs = true);
[areas, boxes] = step(blobAnalysis, imId);

% Sort connected components in descending order by area
[~, Idx] = sort(areas, "Descend");

% Get the two largest components.
boxes = double(boxes(Idx(2), :));

% Reduce the size of the image for display.
scale = magnification / 100;
imDetectedIds = imresize(im, scale);

% Insert labels for the Ids.
imDetectedIds = insertObjectAnnotation(imDetectedIds, "rectangle", ...
    scale * boxes, "Student Id");
figure; imshow(imDetectedIds);
title("Detected Ids");


%%

% Detect the checkerboard.
[imagePoints, boardSize] = detectCheckerboardPoints(im);

% Adjust the imagePoints so that they are expressed in the coordinate system
% used in the original image, before it was undistorted.  This adjustment
% makes it compatible with the cameraParameters object computed for the original image.
imagePoints = imagePoints + newOrigin; % adds newOrigin to every row of imagePoints

% Extract camera intrinsics.
camIntrinsics = cameraParams.Intrinsics;

% Compute extrinsic parameters of the camera.
camExtrinsics = estimateExtrinsics(imagePoints, worldPoints, camIntrinsics);
%%

% Adjust upper left corners of bounding boxes for coordinate system shift 
% caused by undistortImage with output view of 'full'. This would not be
% needed if the output was 'same'. The adjustment makes the points compatible
% with the cameraParameters of the original image.
boxes = boxes + [newOrigin, 0, 0]; % zero padding is added for wIdth and height

% Get the top-left and the top-right corners.
box1 = double(boxes(1, :));
imagePoints1 = [box1(1:2); ...
                box1(1) + box1(3), box1(2)];

% Get the world coordinates of the corners            
worldPoints1 = img2world2d(imagePoints1, camExtrinsics, camIntrinsics);

% Compute the diameter of the Id in millimeters.
lengths = worldPoints1(2, :) - worldPoints1(1, :);
% diameterInMillimeters = hypot(d(1), d(2));
fprintf("Measured length of student Id = %0.2f mm\n", max(lengths));
