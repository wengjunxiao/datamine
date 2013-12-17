clear all;
load 'iris.csv';
meanValue = mean(iris(:,1:4));%����ÿһ�е�ƽ��ֵ
stdValue = std(iris(:,1:4));%����ÿһ�еķ���
irisData = zeros(150,5);
for i = 1:4
    irisData(:,i) = (iris(:,i) - meanValue(i)) / stdValue(i);%ǰ4�й�һ��
end
irisData(:,5) = iris(:,5);%��ֵ�����
trainExample = [irisData(1:ceil(50*2/3),:); irisData(51:50 + ceil(50*2/3),:); irisData(101:100 + ceil(50*2/3),:)];%ȡÿ���ǰ2/3��Ϊѵ����
testExample = [irisData(ceil(50*2/3) + 1:50,:); irisData(51 + ceil(50*2/3):100,:); irisData(101 + ceil(50*2/3):150,:)];%ȡÿ��ĺ�1/3��Ϊ���Լ�
trueClass = testExample(:,5);%�Ѳ��Լ�ԭ����֪������¼����
testExample(:,5) = 0;

k = 10;%k-nearest neighbor
dist_class = zeros(size(trainExample,1),2);%�þ����Ų��Ե㵽ѵ�����ŷ����¾��룬��2�д��ѵ��������
for i = 1:size(testExample,1)
    for j = 1:size(trainExample,1)
        dist_class(j,1) = norm(testExample(i,1:4) - trainExample(j,1:4));
        dist_class(j,2) = trainExample(j,5);    
    end
    [B,IX] = sort(dist_class,1);%�����жԾ������У�B�����кõľ���IX��ԭ�����б�
    mindc = B(1:k,:);%ȡk��������ٽ���
    for ii = 1:k %���´�����Ϊ�˱�֤��2�е����Ҳ���ŵ�1�еľ���任λ�ö��任����Ӧ��λ��,���ҵ�����Ӧ�ľ���
            mindc(ii,2) = dist_class(IX(ii,1),2);
    end
    
    class1_num = size(find(mindc(:,2) == 1),1);%������У���1��ĸ���
    class2_num = size(find(mindc(:,2) == 2),1);%������У���2��ĸ���
    class3_num = size(find(mindc(:,2) == 3),1);%������У���3��ĸ���
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
    fprintf('׼ȷ�ʣ�%f\n',accuracy);