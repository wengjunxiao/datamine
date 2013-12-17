
clear all
clc
close all


TrainDatabasePath = uigetdir('C:\ѧϰ\�ҵĳ���\����о�\PCA', 'Select training database path' );
TestDatabasePath = uigetdir('C:\ѧϰ\�ҵĳ���\����о�\PCA', 'Select test database path');

prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {'1'};

TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.jpg');
im = imread(TestImage);

T = CreateDatabase(TrainDatabasePath);%һ�еľ���
[m, A, Eigenfaces] = EigenfaceCore(T);
OutputName = Recognition(TestImage, m, A, Eigenfaces);

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);

imshow(im)
title('��������');
figure,imshow(SelectedImage);
title('�ҵ���ƥ�������ͼ��');

str = strcat('ƥ�������Ϊ :  ',OutputName);
disp(str)