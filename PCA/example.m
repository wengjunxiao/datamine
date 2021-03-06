
clear all
clc
close all


TrainDatabasePath = uigetdir('C:\学习\我的程序\最近研究\PCA', 'Select training database path' );
TestDatabasePath = uigetdir('C:\学习\我的程序\最近研究\PCA', 'Select test database path');

prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {'1'};

TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.jpg');
im = imread(TestImage);

T = CreateDatabase(TrainDatabasePath);%一列的矩阵
[m, A, Eigenfaces] = EigenfaceCore(T);
OutputName = Recognition(TestImage, m, A, Eigenfaces);

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);

imshow(im)
title('测试人脸');
figure,imshow(SelectedImage);
title('找到的匹配的人脸图像');

str = strcat('匹配的人脸为 :  ',OutputName);
disp(str)
