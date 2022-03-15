clc
clear all
close all
%% 导入医大二院的PH患者和在学校采集的健康心音；
load('E:\2021.11.30程序数据\mine_PH.mat');
%% 将PH患者单独列出来；
for i=1:length(HS_me)
    PH{i,1}=HS_me{i, 1}(:,4);
end
for i=1:length(HS_me)
    PH_segment1{i,1}=PH{i, 1}(1001:3312,1);
end
for i=1:length(PH_segment1)
    % B = filtfilt(a,b,A{1,i});
    PH_segments1{i,1} = PH_segment1{i,1}/ max( abs(PH_segment1{i,1}));%归一化
end
for i=1:length(HS_me)
    PH_segment2{i,1}=PH{i, 1}(10001:12312,1);
end
for i=1:length(PH_segment2)
    % B = filtfilt(a,b,A{1,i});
    PH_segments2{i,1} = PH_segment2{i,1}/ max( abs(PH_segment2{i,1}));%归一化
end
PH_segments=[PH_segments1
    PH_segments2];
PH_segments=PH_segments(1:200,:);
%% 2021.11.24.统一读取数据库中的5种疾病的心音；
maindir='E:\2021.11.30程序数据\Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master\database\'; %指定文件路径
subdir=dir(maindir);%读取该指定文件夹下的多有子文件夹
len=length(subdir);
for i=1:len
    filepath=fullfile(maindir,subdir(i).name,'*.wav');%指定wav文件
    waves=dir(filepath);%wav文件的完整路径，例如：D:\Documents\data\wav\canon\1.wav
    for j=1:length(waves)
        wavepath=fullfile(maindir,subdir(i).name,waves(j).name);
        [x,fs]=audioread(wavepath);%读取音频文件
        data_HS{i,j}=x;
    end
end
%% 时间长度；
% fs = 8000; %信号的采样频率
% dt=1/fs;
% n=length(data_HS{3, 2});
% t=[0:n-1]*dt;

%% 2021.11.30.由于两个数据库的采样频率不同，所以将高采样频率降低;或者对自采的数据进行插值，增强自采数据的点数；
for i=3:7
    for j=1:200
        A{i,j}=resample(data_HS{i, j} ,2000,8000);
    end
end

%% 去掉空集后，将所有cell连接在一起，1000个cell对应1000个labels;
A=A';
idx = find(~cellfun(@isempty,A));
B = arrayfun(@(ii) A{idx(ii)}, [1:1:numel(idx)],'UniformOutput', false);
%% 标签标注：
label_AS=ones(200,1)*1;
label_MR=ones(200,1)*2;
label_MS=ones(200,1)*3;
label_MVP=ones(200,1)*4;
label_N=ones(200,1)*5;
label_PH=ones(200,1)*6;
Labels=[label_AS' label_MR' label_MS' label_MVP' label_N' label_PH'];
Labels=Labels';
%% 数据增强；
% SNR=40;
% for i=1:length(Labels)
%         if Labels(i)==-1
%         aug{i,2}= awgn(data_all{i, 1}(:,2),SNR);
%         end
% end
% idx = find(~cellfun(@isempty,aug));
% aug = arrayfun(@(ii) aug{idx(ii)}, [1:1:numel(idx)],'UniformOutput', false);
% aug=aug';

%% 心音统一预处理；
fs=2000;
f1=20; %cuttoff low frequency to get rid of baseline wander
f2=150; %cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/fs; % cutt off based on fs
N = 3; % order of 3 less processing
[a,b] = butter(N,Wn); %bandpass filtering
for i=1:length(B)
    % B = filtfilt(a,b,A{1,i});
    C{i,1} = B{1,i}/ max( abs(B{1,i}));%归一化
    I(i,1)=length(C{i,1});
end
for i=1:length(B)
    min_len=min(I);
    database_HS{i,:}=C{i,1}(1:min_len,:);
end
database_HS=[database_HS
    PH_segments];
%% 加入高斯噪声增加数据量，然后再对加入高斯噪声的信号进行带通滤波器滤波；（实质上并没有增强信号的多样性）
SNR=10;
sig=database_HS{958, 1};
sig1 = awgn(database_HS{958, 1},SNR,'measured');
sig2 = filtfilt(a,b,sig1);
% figure
% subplot(311);
% plot(sig);
% subplot(312);
% plot(sig1);
% subplot(313);
% plot(sig2);
%% 在原始数据上加入较弱的噪音,效果与加入高斯噪音差不多；
delta=rand(1,1);
pa=randn(1,length(sig))./max(randn(1,length(sig)));
pa=pa';
sig3=sig+delta*pa;
sig4 = filtfilt(a,b,sig3);
% figure
% subplot(311);
% plot(sig);
% subplot(312);
% plot(sig3);
% subplot(313);
% plot(sig4);
%% 在原始的信号上叠加强度更大的噪音，看看此分类性能如何；
pa=randn(1,length(sig))./max(randn(1,length(sig)));
pa=pa';
sig5=sig+pa;
sig6 = filtfilt(a,b,sig5);

fs = 2000; %信号的采样频率
dt=1/fs;
n=length(sig);
t=[0:n-1]*dt;

figure
subplot(311);
plot(t,sig,'color',[0.25 0.41 0.88]);
subplot(312);
plot(t,sig5,'color',[0.24 0.57 0.25]);
subplot(313);
plot(t,sig6,'color',[0.89 0.09 0.05]);
%% 循环生成一倍的更强的噪音数据；
for i=1:1200
    ori_signal=database_HS{i,1};
    noise_signal=randn(1,length(ori_signal))./max(randn(1,length(ori_signal)));
    noise_signal=noise_signal';
    aug_signal=ori_signal+noise_signal;
    augmentation = filtfilt(a,b,aug_signal);
    augmentation_signal{i,:}=augmentation;
end

%% 创建时频表示;
Fs = 2000;
fb = cwtfilterbank('SignalLength',min_len,...
    'SamplingFrequency',Fs,...
    'VoicesPerOctave',12);
sig = database_HS{1000,1};
[cfs,frq] = wt(fb,sig);
t = (0:min_len-1)/Fs;figure;pcolor(t,frq,abs(cfs))
set(gca,'yscale','log');shading interp;axis tight;
title('Scalogram');xlabel('Time (s)');ylabel('Frequency (Hz)')
%% 将尺度图创建为 RGB 图像，并将其写入 dataDir 中的适当子目录。为了与 GoogLeNet 架构兼容，每个 RGB 图像是大小为 224×224×3 的数组。
% database_HS=cell2mat(database_HS');
% database_HS=database_HS';
% data = database_HS;
%% 生成含噪声的图像
% augmentation_signal=cell2mat(augmentation_signal');
% augmentation_signal=augmentation_signal';
% data =augmentation_signal;
% labels = Labels;
% 
% [~,signalLength] = size(data);
% 
% fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
% r = size(data,1);
% 
% for ii = 1:r
%     cfs = abs(fb.wt(data(ii,:)));
%     im = ind2rgb(im2uint8(rescale(cfs)),jet(128));
%     s=strcat('E:\2021.11.30程序数据\2012.12.13.noisy_figures_1200\',num2str(ii+1200),'.jpg');
%     imwrite(imresize(im,[224 224]),s);
% end
% % 再在同一个文件夹内生成没有加噪的图像；
% database_HS=cell2mat(database_HS');
% database_HS=database_HS';
% data = database_HS;
% labels = Labels;
% 
% [~,signalLength] = size(data);
% 
% fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
% r = size(data,1);

% for ii = 1:r
%     cfs = abs(fb.wt(data(ii,:)));
%     im = ind2rgb(im2uint8(rescale(cfs)),jet(128));
%     s=strcat('E:\2021.11.30程序数据\2012.12.13.noisy_figures_1200\',num2str(ii),'.jpg');
%     imwrite(imresize(im,[224 224]),s);
% end

%% 分为训练数据和验证数据;将尺度图图像加载为图像数据存储。
% Labels=[Labels
%     Labels]
% Labels=categorical(Labels);
% Labels=Labels';
% allImages = imageDatastore(fullfile('E:\2021.11.30程序数据\figures\'),...
%     'IncludeSubfolders',true,...
%     'LabelSource','foldernames');

%% 按照自然顺序123，重排allImages.Files路径图片的顺序；
%% 调用sort_net（）函数；
% files = dir('E:\2021.11.30程序数据\figures\');
% files_name =sort_nat({files.name})
% len=length(files);
% for i=1:len
%     oldname=files_name{i};
%     newname=strcat('E:\2021.11.30程序数据\figures\',num2str(i),'.jpg')
%     new_name{i,:}=newname;
%     eval(['!rename' 32 oldname 32  newname]);
% end
% load ('E:\2021.11.30程序数据\3D_CWT\figures_2400.mat');
% allImages.Files=new_name;
% allImages.Labels=Labels;
load('E:\2021.11.30程序数据\3D_CWT\figures_1200_noisy.mat');
load ('E:\2021.11.30程序数据\3D_CWT\figures_1200_clean.mat');
allImages1.Labels=categorical(allImages1.Labels);
allImages.Labels=categorical(allImages.Labels);
% load('E:\2021.11.30程序数据\3D_CWT\allinmages_6_all.mat');
%% 按照比列划分训练集和验证集；
%% 其中，数据增强的部分全部用来训练，未增强的原始数据中80%训练，20%测试；
%% 若划分训练集，验证机，测试集；
% [trainInd,valInd,testInd] = dividerand(1200,0.7,0.15,0.15);
indices = crossvalind('Kfold',allImages.Labels,10);
for j = 1:10
    test = (indices == j);
    train = ~test;
    rng default
    % [imgsTrain1,imgsValidation] = splitEachLabel(allImages,0.8,'randomized');%原始的信号划分出训练集和测试集；
    load ('E:\2021.11.30程序数据\3D_CWT\figures_1200_clean.mat');
    load('E:\2021.11.30程序数据\3D_CWT\figures_1200_noisy.mat');
    allImages1.Labels=categorical(allImages1.Labels);% 数据扩增后的图像；
    allImages.Labels=categorical(allImages.Labels);% 干净的图像，原始的图像，未数据扩增的图像。
    train_data = allImages.Files(train);%不带1的表示数据未扩增的图像，原始干净的图像；
    train_label = allImages.Labels(train);
    test_data = allImages.Files(test);
    test_label = allImages.Labels(test);
    
    %% 训练
    imgsTrain=allImages;
    imgsTrain.Files=[train_data
        allImages1.Files];%训练集中融合了原始的90%的数据+扩增的数据；
    imgsTrain.Labels=[train_label
        allImages1.Labels
        ];
    %     imgsValidation.Files=test_data;
    %     imgsValidation.Labels=test_label;
    
    disp(['Number of training images: ',num2str(numel(imgsTrain.Files))]);
    %% 验证:每次训练和验证都需要统一图像的格式，所以每次都需要导入一个图像格式，再替换里面的数据；
    load ('E:\2021.11.30程序数据\3D_CWT\figures_1200_clean.mat');
    load('E:\2021.11.30程序数据\3D_CWT\figures_1200_noisy.mat');
    allImages1.Labels=categorical(allImages1.Labels);% 数据扩增后的图像；
    allImages.Labels=categorical(allImages.Labels);% 干净的图像，原始的图像，未数据扩增的图像。
    imgsValidation = allImages;
    imgsValidation.Files  =test_data;
    imgsValidation.Labels=test_label;
    disp(['Number of validation images: ',num2str(numel(imgsValidation.Files))]);
    %% 加载Googlenet;
    net = googlenet;
    lgraph = layerGraph(net);
    numberOfLayers = numel(lgraph.Layers);
    figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph)
    title(['GoogLeNet Layer Graph: ',num2str(numberOfLayers),' Layers']);
    net.Layers(1)
    %% 修改 GoogLeNet 网络参数;
    newDropoutLayer = dropoutLayer(0.6,'Name','new_Dropout');
    lgraph = replaceLayer(lgraph,'pool5-drop_7x7_s1',newDropoutLayer);
    
    numClasses = numel(categories(allImages.Labels));
    newConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc',...
        'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
    lgraph = replaceLayer(lgraph,'loss3-classifier',newConnectedLayer);
    
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'output',newClassLayer);
    
    %% 设置训练选项并训练 GoogLeNet;
    options = trainingOptions('sgdm',...
        'MiniBatchSize',10,...
        'MaxEpochs',15,...
        'InitialLearnRate',1e-4,...
        'ValidationData',imgsValidation,...
        'ValidationFrequency',10,...
        'Verbose',1,...
        'ExecutionEnvironment','cpu',...
        'Plots','training-progress');
    rng default
    %     trainedGN = trainNetwork(allImages2,lgraph,options);
    trainedGN = trainNetwork(imgsTrain,lgraph,options);
    trainedGN.Layers(end)
    %% 评估 GoogLeNet 准确度;
    [YPred,probs] = classify(trainedGN,imgsValidation);
    accuracy = mean(YPred==imgsValidation.Labels);
    disp(['GoogLeNet Accuracy: ',num2str(100*accuracy),'%'])
    
    %% 了解 GoogLeNet 激活;
    wghts = trainedGN.Layers(2).Weights;
    wghts = rescale(wghts);
    wghts = imresize(wghts,5);
    figure
    montage(wghts)
    title('First Convolutional Layer Weights')
    
    %% 画出混淆矩阵；
    figure
    confusionMat = confusionmat(imgsValidation.Labels,YPred);
    confusionchart(imgsValidation.Labels,YPred, ...
        'Title',sprintf('Confusion Matrix on Validation (overall accuracy: %.4f)',accuracy),...
        'ColumnSummary','column-normalized','RowSummary','row-normalized');
    %% 评价指标：查准率、召回率、敏感性、特异性和F1-score；
    Num_right=confusionMat(1,1)+confusionMat(2,2)+confusionMat(3,3)+confusionMat(4,4)+confusionMat(5,5)+confusionMat(6,6);
    Num_wrong=numel(imgsValidation.Files)-(confusionMat(1,1)+confusionMat(2,2)+confusionMat(3,3)+confusionMat(4,4)+confusionMat(5,5)+confusionMat(6,6));
    ACC(j,1)=(Num_right)./numel(imgsValidation.Files);
    %% 因为是6分类所以要循环6次，分别计算每一分类的各项指标；i代表从第一类到第六类；
    for i=1:6
        Precision(i,j) = confusionMat(i,i)./sum(confusionMat(:,i));
        Recall(i,j) = confusionMat(i,i)./sum(confusionMat(i,:));
        F1_score(i,j) = 2*confusionMat(i,i)./(sum(confusionMat(:,i))+sum(confusionMat(i,:)));
    end
    
end
%% 结果矩阵；

%% 画出损失Loss和准确性图；
% figure; plot(tinfo.TrainingLoss);
% figure; plot(tinfo.TrainingAccuracy);

%% 画出ROC/AUC曲线面积；
probs=double(probs);
ground_truth=double(ground_truth);
%% 分别画出类1到类6的图，同时plot在一张图上；
ground_truth1=[ground_truth(1:20,1)
    ones(100,1)*0];
ground_truth2=[ones(20,1)*0
    ground_truth(21:40,1)./2
    ones(80,1)*0];
ground_truth3=[ones(40,1)*0
    ground_truth(41:60,1)./3
    ones(60,1)*0];
ground_truth4=[ones(60,1)*0
    ground_truth(61:80,1)./4
    ones(40,1)*0];
ground_truth5=[ones(80,1)*0
    ground_truth(81:100,1)./5
    ones(20,1)*0];
ground_truth6=[ones(100,1)*0
    ground_truth(101:120,1)./6];

%% 画出ROC-AUC曲线；
x = 1.0;
y = 1.0;
%计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num
predict=probs(:,1);
ground_truth=ground_truth1;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%首先对predict中的分类器输出值按照从小到大排列
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end
%画出图像
figure;
plot(X,Y,'Color','[1 0.5 0]','LineWidth',2,'MarkerSize',3);
%计算小矩形的面积,返回auc
auc1 = -trapz(X,Y);

x = 1.0;
y = 1.0;
predict=probs(:,2);
ground_truth=ground_truth2;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%首先对predict中的分类器输出值按照从小到大排列
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end
%画出图像
hold on
plot(X,Y,'r','LineWidth',2,'MarkerSize',3);
%计算小矩形的面积,返回auc
auc2 = -trapz(X,Y);

x = 1.0;
y = 1.0;
predict=probs(:,3);
ground_truth=ground_truth3;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%首先对predict中的分类器输出值按照从小到大排列
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end
%画出图像
hold on
plot(X,Y,'b','LineWidth',2,'MarkerSize',3);
%计算小矩形的面积,返回auc
auc3 = -trapz(X,Y);

x = 1.0;
y = 1.0;
predict=probs(:,4);
ground_truth=ground_truth4;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%首先对predict中的分类器输出值按照从小到大排列
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end
%画出图像
hold on
plot(X,Y,'c','LineWidth',2,'MarkerSize',3);
%计算小矩形的面积,返回auc
auc4 = -trapz(X,Y);

x = 1.0;
y = 1.0;
predict=probs(:,5);
ground_truth=ground_truth5;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%首先对predict中的分类器输出值按照从小到大排列
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end
%画出图像
hold on
plot(X,Y,'g','LineWidth',2,'MarkerSize',3);
%计算小矩形的面积,返回auc
auc5 = -trapz(X,Y);

x = 1.0;
y = 1.0;
predict=probs(:,6);
ground_truth=ground_truth6;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%首先对predict中的分类器输出值按照从小到大排列
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end
%画出图像
hold on
plot(X,Y,'m','LineWidth',2,'MarkerSize',3);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve');
%计算小矩形的面积,返回auc
auc6 = -trapz(X,Y);
legend('AS','MR','MS','MVP','N','PH');

hold on
plot([0 0.1 0.4 0.6 0.8 1],[0 0.1 0.4 0.6 0.8 1],'k--')
AUC=[auc1' auc2' auc3' auc4' auc5' auc6'];



