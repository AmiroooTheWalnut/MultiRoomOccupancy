function [noise]=occNoiseMeasure(occ,minimumLength)
noise=0;
for i=minimumLength:size(occ,2)
    for j=1:minimumLength-1
        arr=occ(1,i-minimumLength+1:i);
        noise=noise+sum(diff(arr)~=0);
        % avg=mean(occ(1,i-minimumLength+1:i));
        % noise=noise+sum(abs(occ(1,i-minimumLength+1:i)-avg));
    end
end