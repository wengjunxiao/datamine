function T = CreateDatabase(TrainDatabasePath)
TrainFiles = dir(TrainDatabasePath);
Train_Number = 0;

for i = 1:size(TrainFiles,1)
    if not(strcmp(TrainFiles(i).name,'.')|strcmp(TrainFiles(i).name,'..'))
        Train_Number = Train_Number + 1; %���е�ͼ������
    end
end
T = [];
for i = 1 : Train_Number
    str = int2str(i);
    str = strcat('\',str,'.jpg');
    str = strcat(TrainDatabasePath,str);
    
    img = imread(str);
    img = rgb2gray(img);
    
    [irow icol] = size(img);
   
    temp = reshape(img',irow*icol,1); %���һ��
    T = [T temp]; 
end