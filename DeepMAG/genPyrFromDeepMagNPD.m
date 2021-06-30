warning('off','all')
addpath(genpath('\\vibe15\PublicAll\t-moali\DeepMagTest\'));
vidInPath = "\\vibe15\PublicAll\t-moali\videos\15fps\256\npd\";
vidOutPath = "\\vibe15\PublicAll\t-moali\videos\15fps\256\npd\deepMagPyr\";

filesDone = dir(vidOutPath)
[numofExistingFiles,~] = size(filesDone);

files = dir(vidInPath);
[numofFiles,~] = size(files);
for i = 1:numofFiles
    
    if files(i).isdir == 0
        process = true;
        for j = 1:numofExistingFiles
            if strcmp(files(i).name+"Mag.mat" , filesDone(j).name)
                process = false;
            end
        end
        if process == true && contains(files(i).name, ".mat")==false && contains(files(i).name, "task1")==false && contains(files(i).name, "task6")==false
            video = vidInPath+files(i).name;
            nnCropping_br_Hand( video );
            
            matFile = video+".mat"; 
            
            pythonCmd = "python \\vibe15\PublicAll\t-moali\DeepMagTest\deepMagPyrGen.py --video "  + matFile + " --out_dir " + vidInPath + "deepMagPyr\" + files(i).name;
            status = system( pythonCmd );
            
        end
    end
end