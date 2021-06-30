vidInPath = "\\vibe15\PublicAll\t-moali\videos\15fps\256\pd\segmented-pd\";

pyrPath = "\\vibe15\PublicAll\t-moali\videos\15fps\256\pd\segmented-pd\pyr\";

files = dir(vidInPath); 
existingFiles = dir("\\vibe15\PublicAll\t-moali\videos\15fps\256\pd\segmented-pd\pyr\")

[numofFiles,~] = size(files);
[numofExistingFiles,~] = size(existingFiles);

%numofFiles = 5;

for i = 1:numofFiles
    if files(i).isdir == 0 && contains(files(i).name,'preview')
        f = files(i).name;
        process = true;
        for j = 1:numofExistingFiles
            if strcmp(f+".mat" , existingFiles(j).name)
                process = false;
            end
        end
        if process == true
            f
            phaseAmplifySimple (vidInPath + f , 4, 0.1, 0.5, vidInPath+"PhaseMag\"+f,pyrPath+f+".mat");
        end
    end
end 