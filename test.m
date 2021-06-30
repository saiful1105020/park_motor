[m,n] = size(quality);

badFiles = 0;
validVideos = 0;
P = zeros(1,n);
Idx = zeros(1,n);
for i=1:n
    if length(quality{i})==0
        badFiles = badFiles + 1;
    else
        validVideos = validVideos+1;
        P(1,validVideos) = quality{i};
        Idx(1,validVideos) = i;
    end
end

sum(isnan(P))
badFiles

P = P(:,1:badFiles);
[v, i] = max(P);
v
fileName{Idx(1,i)}