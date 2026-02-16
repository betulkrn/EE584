% Load the stereoParameters object.
load('handshakeStereoParams.mat');

% Visualize camera extrinsics.
showExtrinsics(stereoParams);


videoFileLeft = 'handshake_left.avi';
videoFileRight = 'handshake_right.avi';

readerLeft = VideoReader(videoFileLeft);
readerRight = VideoReader(videoFileRight);
player = vision.VideoPlayer('Position', [20,200,740 560]);


frameLeft = readFrame(readerLeft);
frameRight = readFrame(readerRight);

[frameLeftRect, frameRightRect, reprojectionMatrix] = ...
    rectifyStereoImages(frameLeft, frameRight, stereoParams);

figure;
imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
title('Rectified Video Frames');



frameLeftGray  = im2gray(frameLeftRect);
frameRightGray = im2gray(frameRightRect);
    
disparityMap = disparitySGM(frameLeftGray, frameRightGray);
figure;
imshow(disparityMap);
title('Disparity Map');
colormap jet
colorbar

%%

points3D = reconstructScene(disparityMap, reprojectionMatrix);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);



% Create the people detector object. Limit the minimum object size for
% speed.
peopleDetector = vision.PeopleDetector('MinSize', [166 83]);

% Detect people.
bboxes = peopleDetector.step(frameLeftGray);


% Find the centroids of detected people.
centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
    round(bboxes(:, 2) + bboxes(:, 4) / 2)];

% Find the 3-D world coordinates of the centroids.
centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
X = points3D(:, :, 1);
Y = points3D(:, :, 2);
Z = points3D(:, :, 3);
centroids3D = [X(centroidsIdx)'; Y(centroidsIdx)'; Z(centroidsIdx)'];

% Find the distances from the camera in meters.
dists = sqrt(sum(centroids3D .^ 2));
    
% Display the detected people and their distances.
labels = cell(1, numel(dists));
for i = 1:numel(dists)
    labels{i} = sprintf('%0.2f meters', dists(i));
end
figure;
imshow(insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes, labels));
title('Detected People');


%%
while hasFrame(readerLeft) && hasFrame(readerRight)
    % Read the frames.
    frameLeft = readFrame(readerLeft);
    frameRight = readFrame(readerRight);
    
    % Rectify the frames.
    [frameLeftRect, frameRightRect] = ...
        rectifyStereoImages(frameLeft, frameRight, stereoParams);
    
    % Convert to grayscale.
    frameLeftGray  = im2gray(frameLeftRect);
    frameRightGray = im2gray(frameRightRect);
    
    % Compute disparity. 
    disparityMap = disparitySGM(frameLeftGray, frameRightGray);
    
    % Reconstruct 3-D scene.
    points3D = reconstructScene(disparityMap, reprojectionMatrix);
    points3D = points3D ./ 1000;
    ptCloud = pointCloud(points3D, 'Color', frameLeftRect);
    view(player3D, ptCloud);
    
    % Detect people.
    bboxes = peopleDetector.step(frameLeftGray);
    
    if ~isempty(bboxes)
        % Find the centroids of detected people.
        centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
            round(bboxes(:, 2) + bboxes(:, 4) / 2)];
        
        % Find the 3-D world coordinates of the centroids.
        centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
        X = points3D(:, :, 1);
        Y = points3D(:, :, 2);
        Z = points3D(:, :, 3);
        centroids3D = [X(centroidsIdx), Y(centroidsIdx), Z(centroidsIdx)];
        
        % Find the distances from the camera in meters.
        dists = sqrt(sum(centroids3D .^ 2, 2));
        
        % Display the detect people and their distances.
        labels = cell(1, numel(dists));
        for i = 1:numel(dists)
            labels{i} = sprintf('%0.2f meters', dists(i));
        end
        dispFrame = insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes,...
            labels);
    else
        dispFrame = frameLeftRect;
    end
    
    % Display the frame.
    step(player, dispFrame);
end

%%
% Clean up
release(player);

