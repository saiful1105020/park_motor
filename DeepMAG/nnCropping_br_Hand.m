function nnCropping_br_Hand(video, outputVideo)
%% classical cardiocam
% clear
% clc
% close all

%% Load parameters
startTime = 0;
endTime = 10;
% subNum = 21;
% taskNum = 3;
camNum = 'B2';
dataURL=[video];

%% Crop images
L = 36;
t = [];
i = 1;
vidObj = VideoReader(dataURL);
vidObj.CurrentTime = startTime;
endTime = floor(vidObj.Duration);
vidFrame = readFrame(vidObj);
t = [t vidObj.CurrentTime];
if vidObj.Height>vidObj.Width,
    vidRGB = im2double(vidFrame(vidObj.Height/2-vidObj.Width/2+1:vidObj.Height/2+vidObj.Width/2,:,:));
    vidRGB = imresize(vidRGB,[492,492]);
else,
    vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
    vidRGB = imresize(vidRGB,[492,492]);
end
%vidRGB = im2double(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:));
vidNTSC = rgb2ntsc(vidRGB);

numLevels = 3;
numOrients = 4;
filt = numOrients-1;
twidth = 1; 
[pyrCA, pind] = buildSCFpyr(vidNTSC(:,:,1), numLevels, filt, twidth);

L = pind(10,1);
idx = [pyrBandIndices(pind,10) pyrBandIndices(pind,11) pyrBandIndices(pind,12) pyrBandIndices(pind,13)];
Xsub = zeros(round(vidObj.FrameRate*(endTime-startTime))+1,L,L,4,'single');
Xsub(i,:,:,:) = reshape(angle(pyrCA(idx)),L,L,4);
i=i+1;

while hasFrame(vidObj) && (vidObj.CurrentTime <= endTime)
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
    
    pyrCA = buildSCFpyr(vidNTSC(:,:,1), numLevels, filt, twidth);    
    
    Xsub(i,:,:,:) = reshape(angle(pyrCA(idx)),L,L,4);
    i=i+1;
end
% figure,imshow(uint8(imresize(vidFrame(:,vidObj.Width/2-vidObj.Height/2+1:vidObj.Width/2+vidObj.Height/2,:),[L,L])));
% figure,plot(unwrap(Xsub(:,round(474/4),240/4)));
% figure,plot(wrapToPi(diff(Xsub(:,round(474/4),240/4))));

%% Ground truth
% load(['\\vibe15\PublicAll\damcduff\RPPG\ncPPG_Study_Data\P' num2str(subNum) '\BIOSEMI\P' num2str(subNum) 'T' num2str(taskNum) '_BioSEMIData.mat'],'Fs','BandPassFilteredPPG','DSRawRespData');
% tPPG = (0:length(BandPassFilteredPPG)-1)/Fs;
% ysub = single(interp1(tPPG,BandPassFilteredPPG,t,'pchip'));
% rsub = single(interp1(tPPG,DSRawRespData,t,'pchip'));
% figure,plot(t,ysub);

%% Difference
dXsub = wrapToPi(diff(Xsub));
% dysub = zeros(1,length(t)-1,'single');
% drsub = zeros(1,length(t)-1,'single');
% for i = 1:length(t)-1
%     dysub(i)=(ysub(i+1)-ysub(i))/(ysub(i+1)+ysub(i));
%     dysub(i)=ysub(i+1)-ysub(i);
%     drsub(i)=rsub(i+1)-rsub(i);
% end
% figure,histogram(dysub);
clear Xsub;

%% Fill NaN
% dXsub(isnan(dXsub)) = 0;

%% Normalization
% sdThr=std(dXsub(:))*3;
% dXsub(dXsub>sdThr)=sdThr;
% dXsub(dXsub<-sdThr)=-sdThr;

sdNew = std(dXsub(:));
% save(['P' num2str(subNum) 'T' num2str(taskNum) 'sdNew_phase.mat'],'sdNew');
dXsub = dXsub/sdNew;
% dysub = dysub/std(dysub);
% drsub = drsub/std(drsub);

% Xsub = Xsub - mean(Xsub(:));
% Xsub = Xsub/std(Xsub(:));

%% Concat
% dXsub = cat(4,dXsub,Xsub(1:end-1,:,:,:));

%% Save
save([outputVideo(1:end-4) '.mat'],'dXsub','sdNew','-v7.3');
%save([video(1:end-4) '.mat'],'dXsub','sdNew','latest');
