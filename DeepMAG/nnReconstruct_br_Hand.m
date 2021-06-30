function nnReconstruct_br_Hand(video, outputVideo)
%% classical cardiocam
% clear
% clc
% close all
%% Load parameters
startTime = 0;
endTime = 10; %18 315
% subNum = 21;
% taskNum = 3;
camNum = 'B2';
dataURL=[video];
%load([video(1:end-4) '.mat']); %feature matrix
load([outputVideo(1:end-4) '.mat']); %feature matrix
load([outputVideo(1:end-4) 'Mag.mat']); %magnified feature matrix
%load(['C:\20171214\data_br\P' num2str(subNum) 'T' num2str(taskNum) 'Video' camNum '_phase.mat']); %feature matrix
%load(['C:\20171214\result_br_indx2\P' num2str(subNum) 'T' num2str(taskNum) 'Video' camNum '_phase.mat']); %magnified feature matrix

%% Crop images
t = [];
i = 1;
vidObj = VideoReader(dataURL);
vidObj.CurrentTime = startTime;

vidFrame = readFrame(vidObj);
t = [t vidObj.CurrentTime];
%vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
if vidObj.Height>vidObj.Width,
    vidRGB = im2double(vidFrame(vidObj.Height/2-vidObj.Width/2+1:vidObj.Height/2+vidObj.Width/2,:,:));
    vidRGB = imresize(vidRGB,[492,492]);
else,
    vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
    vidRGB = imresize(vidRGB,[492,492]);
end
vidNTSC = rgb2ntsc(vidRGB);

numLevels = floor(log2(min(size(vidRGB(:,:,1))))) - 2;
numOrients = 4;
filt = numOrients-1;
twidth = 1; 
[pyrCA, pind] = buildSCFpyr(vidNTSC(:,:,1), numLevels, filt, twidth);

L = pind(10,1);
idx = [pyrBandIndices(pind,10) pyrBandIndices(pind,11) pyrBandIndices(pind,12) pyrBandIndices(pind,13)];
% Xsub = zeros(round(vidObj.FrameRate*(endTime-startTime))+1,L,L,4,'single');
Xsub = reshape(angle(pyrCA(idx)),L,L,4);
% i=i+1;
% 
% while hasFrame(vidObj) && (vidObj.CurrentTime <= endTime)
%     vidFrame = readFrame(vidObj);
%     t = [t vidObj.CurrentTime];
%     vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
%     vidNTSC = rgb2ntsc(vidRGB);    
%     
%     pyrCA = buildSCFpyr(vidNTSC(:,:,1), numLevels, filt, twidth);    
%     
%     Xsub(i,:,:,:) = reshape(angle(pyrCA(idx)),L,L,4);
%     i=i+1
% end
% load('Xsub.mat');
% figure,imshow(uint8(imresize(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:),[L,L])));

%% Difference
% dXsub = wrapToPi(diff(Xsub));

%% Fill NaN
% dXsub(isnan(dXsub)) = 0;

%% Normalization
% sdThr=std(dXsub(:))*3;
% dXsub(dXsub>sdThr)=sdThr;
% dXsub(dXsub<-sdThr)=-sdThr;
% sdNew = std(dXsub(:));
% dXsub = dXsub/std(dXsub(:));
% load(['E:\Research\20171214\P' num2str(subNum) 'T' num2str(taskNum) '\P' num2str(subNum) 'T' num2str(taskNum) 'sdNew_phase.mat']);

%% detrending
% Xold = dXsub/sdNew;
% Xnew = Xnew*10;

% Xnew(:,:,:,1) = Xnew(:,:,:,1) - mean(mean(mean(Xnew(:,:,:,1),1),2),3);
% Xnew(:,:,:,2) = Xnew(:,:,:,2) - mean(mean(mean(Xnew(:,:,:,2),1),2),3);
% Xnew(:,:,:,3) = Xnew(:,:,:,3) - mean(mean(mean(Xnew(:,:,:,3),1),2),3);
% Xnew = Xnew - mean(Xnew);
% Xnew = Xnew - movmean(Xnew, 171);
% Xnew = Xnew - movmean(Xnew, 751) + movmean(Xold, 751);
% Xnew = Xold - movmean(Xold, 241) + movmean(Xnew, 241) - movmean(Xnew, 751) + movmean(Xold, 751);
% Xnew = Xold - movmean(Xold, 441) + movmean(Xnew, 441) - movmean(Xnew, 551) + movmean(Xold, 551);
% fs = 120;
% [b,a] = butter(3,[0.08/fs*2 0.5/fs*2]);
% Xnew = Xold + 10*filtfilt(b,a,double(Xold));

%% denormalization
% dXsubnew = Xnew*sdNew;
% dXsubnew = dXsub;

% figure,plot((1:3600)/120,cumsum(dXsub(:,108,94,3))*10)
% hold on,plot((1:3600)/120,cumsum(dXsubnew(:,108,94,3)-mean(dXsubnew(:,108,94,3))))
% xlabel('Time/s'),ylabel('Phase');
% figure,plot((1:3600)/120,cumsum(dXsub(:,104,94,3))*10)
% hold on,plot((1:3600)/120,cumsum(dXsubnew(:,104,94,3)-mean(dXsubnew(:,104,94,3))))
% xlabel('Time/s'),ylabel('Phase');

%% reconstruct
Xsubnew = zeros(size(Xnew,1)+1,L,L,4,'single');
Xsubnew(1,:,:,:) = Xsub;
Xsubnew(2:end,:,:,:) = Xnew*sdNew;
Xsubnew = cumsum(Xsubnew);
% clear dXsubnew Xnew
Xsubold = zeros(size(Xnew,1)+1,L,L,4,'single');
Xsubold(1,:,:,:) = Xsub;
%Xsubold(1:end,:,:,:) = dXsub*sdNew;
Xsubold = dXsub*sdNew;
Xsubold = cumsum(Xsubold);

%% generate video - 36x36
% vidObj = VideoWriter('result.avi');
% vidObj.FrameRate = 120;
% open(vidObj);
% for i = 1:size(Xnew,1)+1
%     writeVideo(vidObj,im2uint8(imresize(squeeze(Xsubnew(i,:,:,:)),10)));
% end
% close(vidObj);

%%
fs = 15;
[b,a] = butter(3,[1/fs*2 5/fs*2]);
% Xsubdiff1 = 10*filtfilt(b,a,double(unwrap(Xsub)));
Xsubdiff = filtfilt(b,a,double(unwrap(Xsubnew)));
% Xsubdiff2 = - movmean(unwrap(Xsub), 241) + movmean(unwrap(Xsubnew), 241) - movmean(unwrap(Xsubnew), 751) + movmean(unwrap(Xsub), 751);
% Xsubdiff = Xsubnew - Xsub;

%% fix phase
% cphase = zeros(123,123,4);
% for i=1:123
%     for j=1:123
%         for k=1:4
%             ctemp=corrcoef(Xsubdiff2(:,108,94,k),Xsubdiff2(:,i,j,k));
%             cphase(i,j,k) = ctemp(1,2);
%         end
%     end
% end
% cphase(:,:,2)=-cphase(:,:,2);
% cphase(:,:,4)=-cphase(:,:,4);
% Xsubdiff = Xsubdiff2.*reshape((cphase>0)*2-1,1,123,123,4);

%% fix phase 2
% cphase = zeros(123,123,4);
% for i=1:123
%     for j=1:123
%         for k=1:4
%             ctemp=corrcoef(cumsum(dXsub(:,i,j,k)),Xsubdiff2(2:end,i,j,k));
%             cphase(i,j,k) = ctemp(1,2);
%         end
%     end
% end
% Xsubdiff = Xsubdiff2.*reshape((cphase>0)*2-1,1,123,123,4);

%% clipping
% Xsubdiff(Xsubdiff>2*pi)=0;
% Xsubdiff(Xsubdiff<-2*pi)=0;
Xmax = max(unwrap(Xsubold));
Xmin = min(unwrap(Xsubold));

%Xsubdiff=Xsubdiff.*(Xmax<2*pi).*(Xmin>-2*pi);

%% generate video - full
Q = 2;
%vidObj2 = VideoWriter([video(1:end-4) 'Mag.avi']);
vidObj2 = VideoWriter([outputVideo(1:end-4) 'Mag.avi'])
vidObj2.FrameRate = 15;
open(vidObj2);
vidObj.CurrentTime = startTime;
for i = 1:size(Xnew,1)+1
    vidFrame = readFrame(vidObj);
    %vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
    if vidObj.Height>vidObj.Width,
        vidRGB = im2double(vidFrame(vidObj.Height/2-vidObj.Width/2+1:vidObj.Height/2+vidObj.Width/2,:,:));
        vidRGB = imresize(vidRGB,[492,492]);
    else,
        vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
        vidRGB = imresize(vidRGB,[492,492]);
    end
    vidNTSC = rgb2ntsc(vidRGB);    
    
    pyrCA = buildSCFpyr(vidNTSC(:,:,1), numLevels, filt, twidth);
    
    phaseDiff = exp(1i*Xsubdiff(i,:,:,:));    
    pyrCA(idx) = pyrCA(idx).*phaseDiff(:);
    
    idx2 = [pyrBandIndices(pind,14) pyrBandIndices(pind,15) pyrBandIndices(pind,16) pyrBandIndices(pind,17)];
    phaseDiff2 = exp(1i*imresize(squeeze(Xsubdiff(i,:,:,:)),1/2)/Q);
    pyrCA(idx2) = pyrCA(idx2).*phaseDiff2(:);
     
    idx3 = [pyrBandIndices(pind,18) pyrBandIndices(pind,19) pyrBandIndices(pind,20) pyrBandIndices(pind,21)];
    phaseDiff3 = exp(1i*imresize(squeeze(Xsubdiff(i,:,:,:)),1/4)/Q/Q);
    pyrCA(idx3) = pyrCA(idx3).*phaseDiff3(:);

    idx4 = [pyrBandIndices(pind,22) pyrBandIndices(pind,23) pyrBandIndices(pind,24) pyrBandIndices(pind,25)];
    phaseDiff4 = exp(1i*imresize(squeeze(Xsubdiff(i,:,:,:)),1/8)/Q/Q/Q);
    pyrCA(idx4) = pyrCA(idx4).*phaseDiff4(:);

    idx5 = [pyrBandIndices(pind,6) pyrBandIndices(pind,7) pyrBandIndices(pind,8) pyrBandIndices(pind,9)];
    phaseDiff5 = exp(1i*imresize(squeeze(Xsubdiff(i,:,:,:)),2)*Q);
    pyrCA(idx5) = pyrCA(idx5).*phaseDiff5(:);
     
    idx6 = [pyrBandIndices(pind,2) pyrBandIndices(pind,3) pyrBandIndices(pind,4) pyrBandIndices(pind,5)];
    phaseDiff6 = exp(1i*imresize(squeeze(Xsubdiff(i,:,:,:)),4)*Q*Q);
    pyrCA(idx6) = pyrCA(idx6).*phaseDiff6(:);
    
    vidNTSC(:,:,1) = reconSCFpyr(pyrCA, pind, 'all', 'all', twidth);
    
    
%     pyrCA = buildSCFpyr(vidNTSC(:,:,2), numLevels, filt, twidth);    
%     pyrCA(idx) = pyrCA(idx).*phaseDiff(:);
%     pyrCA(idx2) = pyrCA(idx2).*phaseDiff2(:);
%     pyrCA(idx3) = pyrCA(idx3).*phaseDiff3(:);
%     vidNTSC(:,:,2) = reconSCFpyr(pyrCA, pind, 'all', 'all', twidth);
%     
%     pyrCA = buildSCFpyr(vidNTSC(:,:,3), numLevels, filt, twidth);    
%     pyrCA(idx) = pyrCA(idx).*phaseDiff(:);
%     pyrCA(idx2) = pyrCA(idx2).*phaseDiff2(:);
%     pyrCA(idx3) = pyrCA(idx3).*phaseDiff3(:);    
%     vidNTSC(:,:,3) = reconSCFpyr(pyrCA, pind, 'all', 'all', twidth);
    
    writeVideo(vidObj2,im2uint8(ntsc2rgb(vidNTSC)));
    %i
end
close(vidObj2);

%% viz
% vidObj2 = VideoWriter('result.avi');
% vidObj2.FrameRate = 120;
% open(vidObj2);
% map = colormap(parula(256));
% vidObj.CurrentTime = startTime;
% for i = 1:1800
%     vidFrame = readFrame(vidObj);
%     vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
%     vidNTSC = rgb2ntsc(vidRGB);    
%     
%     pyrCA = buildSCFpyr(vidNTSC(:,:,1), numLevels, filt, twidth);
%        
%     vidPhase = reshape(pyrCA(idx),123,123,4);
%     
%     writeVideo(vidObj2,ind2rgb(im2uint8(squeeze(angle(vidPhase(:,:,3)))),map));
%     i
% end
% close(vidObj2);

%% spatial gradient
% vidPhase1 = angle(reshape(pyrCA(idx),123,123,4));
% vidPhase2 = angle(reshape(pyrCA(idx2),62,62,4));
% vidPhase3 = angle(reshape(pyrCA(idx3),31,31,4));
% vidPhase4 = angle(reshape(pyrCA(idx4),16,16,4));
% vidPhase5 = angle(reshape(pyrCA(idx5),246,246,4));
% vidPhase6 = angle(reshape(pyrCA(idx6),492,492,4));
% 
% vidGradient1 = diff(vidPhase1(:,:,3));
% vidGradient2 = diff(vidPhase2(:,:,3));
% vidGradient3 = diff(vidPhase3(:,:,3));
% vidGradient4 = diff(vidPhase4(:,:,3));
% vidGradient5 = diff(vidPhase5(:,:,3));
% vidGradient6 = diff(vidPhase6(:,:,3));
% 
% mean(vidGradient1(vidGradient1>0))
% mean(vidGradient2(vidGradient2>0))
% mean(vidGradient3(vidGradient3>0))
% mean(vidGradient4(vidGradient4>0))
% mean(vidGradient5(vidGradient5>0))
% mean(vidGradient6(vidGradient6>0))



