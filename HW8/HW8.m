clc;close all;clear;
red= imread("redflower.jpg");
figure
imshow(red)


numColors = 3;
L = imsegkmeans(red,numColors);
B = labeloverlay(red,L);
figure
imshow(B)
title("Red flower - Labeled Image RGB")


lab_red = rgb2lab(red);


ab = lab_red(:,:,2:3);
ab = im2single(ab);
pixel_labels = imsegkmeans(ab,numColors,NumAttempts=3);

B2 = labeloverlay(red,pixel_labels);
figure
imshow(B2)
title("Red flower - Labeled Image a*b*")


mask1 = pixel_labels == 1;
cluster1 = red.*uint8(mask1);
figure
imshow(cluster1)
title("Red flower - Objects in Cluster 1");

mask2 = pixel_labels == 2;
cluster2 = red.*uint8(mask2);
figure
imshow(cluster2)
title("Red flower - Objects in Cluster 2");

mask3 = pixel_labels == 3;
cluster3 = red.*uint8(mask3);
figure
imshow(cluster3)
title("Red flower - Objects in Cluster 3");


L = lab_red(:,:,1);
L_blue = L.*double(mask3);
L_blue = rescale(L_blue);
idx_light_blue = imbinarize(nonzeros(L_blue));

blue_idx = find(mask3);
mask_dark_blue = mask3;
mask_dark_blue(blue_idx(idx_light_blue)) = 0;

blue = red.*uint8(mask_dark_blue);
figure
imshow(blue)

clc;clear;
blue= imread("blueflower.jpg");
figure
imshow(blue)


numColors = 3;
L = imsegkmeans(blue,numColors);
B = labeloverlay(blue,L);
figure
imshow(B)
title("Blue flower - Labeled Image RGB")


lab_blue = rgb2lab(blue);


ab = lab_blue(:,:,2:3);
ab = im2single(ab);
pixel_labels = imsegkmeans(ab,numColors,NumAttempts=3);

B2 = labeloverlay(blue,pixel_labels);
figure
imshow(B2)
title("Blue flower - Labeled Image a*b*")


mask1 = pixel_labels == 1;
cluster1 = blue.*uint8(mask1);
figure
imshow(cluster1)
title("Blue flower - Objects in Cluster 1");

mask2 = pixel_labels == 2;
cluster2 = blue.*uint8(mask2);
figure
imshow(cluster2)
title("Blue flower - Objects in Cluster 2");

mask3 = pixel_labels == 3;
cluster3 = blue.*uint8(mask3);
figure
imshow(cluster3)
title("Blue flower - Objects iman Cluster 3");


L = lab_blue(:,:,1);
L_blue = L.*double(mask3);
L_blue = rescale(L_blue);
idx_light_blue = imbinarize(nonzeros(L_blue));

blue_idx = find(mask3);
mask_dark_blue = mask3;
mask_dark_blue(blue_idx(idx_light_blue)) = 0;

blue_last = blue.*uint8(mask_dark_blue);
figure
imshow(blue_last)


