clear all;
load 'iris.csv';
meanValue = mean(iris(:,1:4));%保存每一列的平均值
stdValue = std(iris(:,1:4));%保存每一列的方差
irisData = zeros(150,5);
for i = 1:4
    irisData(:,i) = (iris(:,i) - meanValue(i)) / stdValue(i);%前4列归一化
end
irisData(:,5) = iris(:,5);%赋值好类标
trainExample = [irisData(1:ceil(50*2/3),:); irisData(51:50 + ceil(50*2/3),:); irisData(101:100 + ceil(50*2/3),:)];%取每类的前2/3作为训练集
testExample = [irisData(ceil(50*2/3) + 1:50,:); irisData(51 + ceil(50*2/3):100,:); irisData(101 + ceil(50*2/3):150,:)];%取每类的后1/3作为测试集
trueClass = testExample(:,5);%把测试集原先已知的类标记录下来
testExample(:,5) = 0;

k = 10;%k-nearest neighbor
dist_class = zeros(size(trainExample,1),2);%该矩阵存放测试点到训练点的欧几里德距离，第2列存放训练点的类标
for i = 1:size(testExample,1)
    for j = 1:size(trainExample,1)
        dist_class(j,1) = norm(testExample(i,1:4) - trainExample(j,1:4));
        dist_class(j,2) = trainExample(j,5);    
    end
    [B,IX] = sort(dist_class,1);%根据行对距离排列，B存排列好的矩阵，IX存原本的列标
    mindc = B(1:k,:);%取k个最近的临近点
    for ii = 1:k %以下处理是为了保证第2列的类标也随着第1列的距离变换位置而变换到相应的位置,即找到它对应的距离
            mindc(ii,2) = dist_class(IX(ii,1),2);
    end
    
    class1_num = size(find(mindc(:,2) == 1),1);%最近点中，第1类的个数
    class2_num = size(find(mindc(:,2) == 2),1);%最近点中，第2类的个数
    class3_num = size(find(mindc(:,2) == 3),1);%最近点中，第3类的个数
    if (max([class1_num,class2_num,class3_num]) == class1_num)
        testExample(i,5) = 1;
        fprintf('NO.%d test example: class is 1\n',i);
    elseif (max([class1_num,class2_num,class3_num]) == class2_num)
        testExample(i,5) = 2;
        fprintf('NO.%d test example: class is 2\n',i);
    else
        testExample(i,5) = 3;
        fprintf('NO.%d test example: class is 3\n',i);
    end
end
accurateNum = 0;
    for i = 1:size(trueClass,1)
        if (testExample(i,5) == trueClass(i))
            accurateNum = accurateNum + 1;
        end
    end
    accuracy = accurateNum / size(trueClass,1);
    fprintf('准确率：%f\n',accuracy);