clear; close all; clc;
%Add path of the code directory
addpath(genpath('E:/Saiful/park_motor/DeepMAG'));

%%Run DEEPMAG for the left hand finger tapping videos
% Specify the folder where the input files live.
myFolder = 'E:/Saiful/park_motor/LeftRight/Task2_Videos/LeftBodyPix';
% Output files (temporary mat files and magnified video) will be here
outputFolder = 'E:/Saiful/park_motor/LeftRight/Task2_Videos/LeftDeepMagOutput/';

%if ~isfolder(outputFolder)
%    mkdir outputFolder
%end

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
for k = 1 : length(theFiles)
    try
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        outputFileName = append(outputFolder,baseFileName);
        
        if isfile([outputFileName(1:end-4) 'Mag.avi'])
            continue
        end
        
        disp(outputFileName);
        
        nnCropping_br_Hand(fullFileName, outputFileName);
        %Output mat file after cropping
        matFile = [outputFileName(1:end-4) '.mat']
        %Run DeepMAG using Python
        pythonCmd = ['C:/Users/Rochester/Anaconda3/envs/park_motor/python E:/Saiful/park_motor/DeepMAG/deepMagPyrGen.py --video ' matFile];
        status = system( pythonCmd );
        %Reconstruct the video with magnified motion
        nnReconstruct_br_Hand(fullFileName, outputFileName);
        %Success: display file name
        disp("Success");

        %Delete temporary mat files
        delete(sprintf('%s',matFile))
        magMatFile = [outputFileName(1:end-4) 'Mag.mat'];
        delete(sprintf('%s',magMatFile))
    catch
        errorFileName = theFiles(k).name;
        warning('Error processing file');
        warning(errorFileName);
        errorFiles = [errorFileName errorFiles];
    end
    
end

%Save error logs
errorLogFile = append(outputFolder,'errors.txt');
filePh = fopen(errorLogFile,'w');
fprintf(filePh,'%s\n',errorFiles(:));
fclose(filePh);


%%Run DEEPMAG for the right hand finger tapping videos
% Specify the folder where the input files live.
myFolder = 'E:/Saiful/park_motor/LeftRight/Task2_Videos/RightBodyPix';
% Output files (temporary mat files and magnified video) will be here
outputFolder = 'E:/Saiful/park_motor/LeftRight/Task2_Videos/RightDeepMagOutput/';

%if ~isfolder(outputFolder)
%    mkdir outputFolder
%end

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
for k = 1 : length(theFiles)
    try
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        outputFileName = append(outputFolder,baseFileName);
        
        if isfile([outputFileName(1:end-4) 'Mag.avi'])
            continue
        end
        
        disp(outputFileName);
        
        nnCropping_br_Hand(fullFileName, outputFileName);
        %Output mat file after cropping
        matFile = [outputFileName(1:end-4) '.mat']
        %Run DeepMAG using Python
        pythonCmd = ['C:/Users/Rochester/Anaconda3/envs/park_motor/python E:/Saiful/park_motor/DeepMAG/deepMagPyrGen.py --video ' matFile];
        status = system( pythonCmd );
        %Reconstruct the video with magnified motion
        nnReconstruct_br_Hand(fullFileName, outputFileName);
        %Success: display file name
        disp("Success");

        %Delete temporary mat files
        delete(sprintf('%s',matFile))
        magMatFile = [outputFileName(1:end-4) 'Mag.mat'];
        delete(sprintf('%s',magMatFile))
        
    catch
        errorFileName = theFiles(k).name;
        warning('Error processing file');
        warning(errorFileName);
        errorFiles = [errorFileName errorFiles];
    end
    
end

%Save error logs
errorLogFile = append(outputFolder,'errors.txt');
filePh = fopen(errorLogFile,'w');
fprintf(filePh,'%s\n',errorFiles(:));
fclose(filePh);