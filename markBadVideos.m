vidInPath = "E:\Wasif\PD Motor Feature Extraction\Task2_15_resized_mp4\";
files = dir(vidInPath);
[numofFiles,~] = size(files);
sample_size = 50;

%quality1 = zeros(1,numofFiles);
%fileName1 = strings(1,numofFiles);

for i = 1:numofFiles
    sum_brisqueI = 0;
    if files(i).isdir == 0
        
        dataURL = vidInPath+files(i).name;
        %dataURL
        try
            fileName{i} = files(i).name;
            vidObj = VideoReader(dataURL);
            samples = floor(linspace(0,vidObj.Duration,sample_size));
            for s = 1:sample_size
                vidObj.CurrentTime = samples(s);
                vidFrame = readFrame(vidObj);
                sum_brisqueI = sum_brisqueI + brisque(vidFrame);
            end
            i
            quality{i} = sum_brisqueI/100;
        
            %quality1(i) = sum_brisqueI/100;
            %fileName1(i) = files(i).name;
        catch E
            files(i).name
            disp(E);
        end
    end
end

save(['E:\Wasif\PD Motor Feature Extraction\FingerDetectionOutput\vidQualityResized.mat'],'fileName','quality','-v7.3');