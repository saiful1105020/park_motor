addpath(genpath('\\vibe15\PublicAll\t-moali\DeepMagTest\'));
vidInPath = "\\vibe15\PublicAll\t-moali\DeepMagTest\VideosForPaper\";


files = dir(vidInPath);
[numofFiles,~] = size(files);

for i=1:numofFiles
    if files(i).isdir()==0 && contains(files(i).name, ".mat")==false
        video = vidInPath+files(i).name; 
        nnReconstruct_br_Hand2( video );
    end
end

