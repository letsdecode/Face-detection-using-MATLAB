%% imread
im=imread('1.jpg');
imshow(im);
%show the image using imshow function from plots tab

%% using uigetfile
[file,path]=uigetfile('*.*','Select image');
loc=strcat(path,file);
pic=imread(loc);
imshow(pic);

%% rgb2gray, bw, warning off
warning('off');
imgray=rgb2gray(pic);
figure,imshow(imgray);
le=graythresh(pic);
imbw=im2bw(pic,le);
imshow(imbw);

imbw2=imbinarize(imgray,le);
% imshow(imbw);
%% imwrite
imwrite(imbw2,'bw.jpg','jpg');


%% crop
cpic=imcrop(pic,[100 100 400 400]);
imshow(cpic);

%% resize
rpic=imresize(pic,0.1);
rpic=imresize(pic,[500 500]);
imshow(rpic);
%% flip 
fpic=flip(pic,1);
imshow(pic);
figure,imshow(fpic);
%% rotate
ropic=imrotate(pic,30,'crop');
imshow(ropic);


%% face detect
[file,path]=uigetfile('*.*','Select image');
loc=strcat(path,file);
pic=imread(loc);
pic2=rgb2gray(pic);
%face
ff=vision.CascadeObjectDetector();
bbox=step(ff,pic2);
dd=insertObjectAnnotation(pic,'Rectangle',bbox,'Face');
pts=detectMinEigenFeatures(pic2,'ROI',bbox);
dd = insertMarker(dd,pts,'+','color','green');
imshow(dd);hold on
plot(pts.Location(:,1),pts.Location(:,2));
hold off

%% mouth
fm=vision.CascadeObjectDetector('Mouth');
fm.MergeThreshold=110;
bbox=step(fm,pic2);
dd=insertObjectAnnotation(pic,'Rectangle',bbox,'Mouth');
imshow(dd);
%% Nose
fn=vision.CascadeObjectDetector('Nose');
bbox=step(fn,pic2);
dd=insertObjectAnnotation(pic,'Rectangle',bbox,'Nose');
imshow(dd);

%% Eye pair
fe=vision.CascadeObjectDetector('RightEye','MergeThreshold',40);
bbox=step(fe,pic2);
dd=insertObjectAnnotation(pic,'Rectangle',bbox,'Eye');
imshow(dd);

%% Upper body
fb=vision.CascadeObjectDetector('UpperBody','MergeThreshold',5);
bbox=step(fb,pic2);
dd=insertObjectAnnotation(pic,'Rectangle',bbox,'Body');
imshow(dd);

%% Webcam
web=webcam('HD WebCam');
%preview(web);
% pause(2);
% pp=snapshot(web);
% imshow(pp);
% pause(2);
% clear('web');
while true
    pic=snapshot(web);
    ff=imshow(pic);
    pause(0.01);
end

%% Real-time face detection
clc;close all;
% clear('li');
li=webcam();
im=snapshot(li);
dete=vision.CascadeObjectDetector('Mouth','MergeThreshold',100);
pp=imshow(im);
while true
    im=snapshot(li);
    im2=rgb2gray(im);
    bb=step(dete,im2);
    im2=insertObjectAnnotation(im,'rectangle',bb,'Face');
    imshow(im2);
end


