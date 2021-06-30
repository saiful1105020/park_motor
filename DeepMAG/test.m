clear; close all; clc;
%Add path of the code directory
addpath(genpath('C:/Wasif/PDMotorFeatureExtraction/DeepMAG'));

% Specify the folder where the input files live.
myFolder = 'C:/Wasif/PDMotorFeatureExtraction/BodyPixOutput';
% Output files (temporary mat files and magnified video) will be here
outputFolder = 'E:/Wasif/PDMotorFeatureExtraction/DeepMAGOutput/';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); % Ask for a new one.
    if myFolder == 0
         % User clicked Cancel
         return;
    end
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.mp4'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
%Log the files that were not converted
errorFiles = [];
numErrorFiles = 0;
for k = 1 : length(theFiles)
    try
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        outputFileName = append(outputFolder,baseFileName);
        
        if isfile([outputFileName(1:end-4) 'Mag.avi'])
            continue
        end
        
        numErrorFiles = numErrorFiles +1;
        disp(outputFileName);
    catch
        warning('Error processing file');
    end
    
end

numErrorFiles