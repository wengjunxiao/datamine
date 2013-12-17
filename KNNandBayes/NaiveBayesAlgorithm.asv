load 'iris.csv';
trainExample = [iris(1:ceil(50*2/3),:); iris(51:50 + ceil(50*2/3),:); iris(101:100 + ceil(50*2/3),:)];
testExample = [iris(ceil(50*2/3) + 1:50,:); iris(51 + ceil(50*2/3):100,:); iris(101 + ceil(50*2/3):150,:)];
trueClass = testExample(:,5);%把测试集原先已知的类标记录下来
testExample(:,5) = 0;

%找到每一类的数量放到矩阵之中
rows1 = find(trainExample(:,5) == 1);
rows2 = find(trainExample(:,5) == 2);
rows3 = find(trainExample(:,5) == 3);

%求得每一列的均值和标准差
meanValue1 = mean(trainExample(rows1,1:4));
stdValue1 = std(trainExample(rows1,1:4));
meanValue2 = mean(trainExample(rows2,1:4));
stdValue2 = std(trainExample(rows2,1:4));
meanValue3 = mean(trainExample(rows3,1:4));
stdValue3 = std(trainExample(rows3,1:4));

%得到每一类在训练集中占的比例
pClass1 = size(rows1,1) / size(trainExample,1);
pClass2 = size(rows2,1) / size(trainExample,1);
pClass3 = size(rows3,1) / size(trainExample,1);

for i = 1:size(testExample,1)
    attribute = zeros(1,4);
    ppClass = zeros(3,4);
    for j = 1:4
        attribute(j) = testExample(i,j);
        %类是1,2,3时，各属性值的条件概率
        ppClass(1,j) = 1/(sqrt(2*pi)*stdValue1(j)) * exp(-(attribute(j) - meanValue1(j))^2 / (2*stdValue1(j)^2));
        ppClass(2,j) = 1/(sqrt(2*pi)*stdValue2(j)) * exp(-(attribute(j) - meanValue2(j))^2 / (2*stdValue2(j)^2));
        ppClass(3,j) = 1/(sqrt(2*pi)*stdValue3(j)) * exp(-(attribute(j) - meanValue3(j))^2 / (2*stdValue3(j)^2));
    end
    %各分类，整个属性集的条件概率
    pXC1 = 1;
    pXC2 = 1;
    pXC3 = 1;
    for j = 1:4
        pXC1 = pXC1 * ppClass(1,j);
        pXC2 = pXC2 * ppClass(2,j);
        pXC3 = pXC3 * ppClass(3,j);
    end
    if (max([pXC1*pClass1, pXC2*pClass2, pXC3*pClass3]) == pXC1*pClass1)
        testExample(i,5) = 1;
        fprintf('NO.%d test example: class is 1\n',i);
    elseif (max([pXC1*pClass1, pXC2*pClass2, pXC3*pClass3]) == pXC2*pClass2)
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