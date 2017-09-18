function[denseFeatures]= densifyFeatures(imsegs,features)
denseFeatures=zeros(imsegs.imsize(1),imsegs.imsize(2),44);
for i=1:imsegs.nseg
    [r,c]=find(imsegs.segimage==i);
    %uprintf(sprintf('\nDensifying i=%d #pixels=%d',i,numel(r)));
    for rr=1:numel(r)
        denseFeatures(r(rr),c(rr),:)=features(i,:);
    end
end
