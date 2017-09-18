function fStackIID(fStackPath, resDir, gamma3, tol)
%% global params
expName = 'fStackIID';
if nargin<2
    resDir = fullfile(fStackPath,'results');
    gamma3 = 0.1;
    tol = 1.0;
end
if nargin<3
    gamma3 = 0.1;
    tol = 1.0;
end
if nargin<4
    tol = 1.0;
end

if ~exist(resDir,'dir')
    mkdir(resDir);
end
if ~exist('sig_c','var')
    sig_c = 0.0001;
end
if ~exist('sig_i','var')
    sig_i = 0.8;
end
if ~exist('sig_n','var')
    sig_n = 0.5;
end
g_size = 12;

%% Reading inputs
[~, imName, ~] = fileparts(fStackPath);
F = strsplit(fStackPath,filesep);
if isempty( F{end})
    imName = F{end-1};
else
    imName = F{end};
end

fStackDir = dir(fullfile(fStackPath,'*.jpg'));
% fStackDir = dir(fullfile(fStackPath,'*.png'));
% Sorting fStackDir
for ff = 1:numel(fStackDir)
    t = fStackDir(ff).name;
    tt = strsplit(t,'.');
    fStackNames{ff} = tt{1};
end
[~, idx] = sort(fStackNames);
fStackDir_ = fStackDir;
for i=1:numel(fStackDir)
    fStackDir(i) = fStackDir_(idx(i));
end

for ff = 1:numel(fStackDir)
    fStack{ff} = im2double(imread(fullfile(fStackPath, fStackDir(ff).name)));
end

% computing fStack pMaps & allFocus
fprintf('============== Processing pMaps ==============\n');
[I, fMap] = getFocusPmaps(fStack, gamma3);
[h,w,d] = size(I);

% computing textureLess image
S = RollingGuidanceFilter(I, 3, 0.1, 4); % Requires double image format

% resizing images for grid multiple
hn = g_size*floor(size(I,1)/g_size);
wn = g_size*floor(size(I,2)/g_size);
I = imresize(I,[ hn wn ]);
S = imresize(S,[ hn wn ]);
fMap = imresize(fMap,[ hn wn ]);

x = (1:hn)/hn;
xMap = repmat(x,[wn 1])';
y = (1:wn)/wn;
yMap = repmat(y,[hn 1]);
fMapXY = cat(3,fMap,xMap);
fMapXY = cat(3,fMapXY,yMap);

Nn = hn*wn;

%% LLE Sparse Neighbourhood Constraints

var_pad = 2; var_patch = 5;

imsegs = im2superpixels_1(S,imName);
GCfeatures = getGCfeatures(I,imsegs);
normGCfeatures=cell2mat(arrayfun(@(x) normalizeGCFeatures(GCfeatures(x,:)), [1:1:imsegs.nseg], 'UniformOutput', false ))';    % Normalize GC features
normGCmap= densifyFeatures(imsegs,normGCfeatures);  % Dense GC features

% global computation
varianceOfGCMap=zeros(1,Nn);
for ii=1:size(normGCmap,3)
    varianceOfGCMap = varianceOfGCMap + var(im2col(padarray(normGCmap(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
end
vGCMap = reshape(varianceOfGCMap, [hn wn]);
% mGCMap = permute(vGCMap, [2 1 3]);
% nNeighbors = getGridLLEMatrixFeatures_Global_tol(normGCmap, mGCMap, 50, size(normGCmap,3), g_size, tol); % Global neighbourhoods constraints


% %global computation
configPath = './libs/rp-master/config/rp_4segs.mat';
config = LoadConfigFile(configPath);
proposals = RP(im2uint8(I), config);
cmins = proposals(:,1);
rmins = proposals(:,2);
cmaxs = proposals(:,3);
rmaxs = proposals(:,4);
h = rmaxs - rmins;
w = cmaxs - cmins;
idx = (h<g_size | w<g_size);
proposals(idx,:) = [];
proposals = proposals';

% varianceOfFMap=zeros(1,Nn);
% for ii=1:size(fMap,3)
%     varianceOfFMap = varianceOfFMap + var(im2col(padarray(fMap(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
% end
% vMap = reshape(varianceOfFMap, [hn wn]);


varOfProposals = zeros(1,size(proposals,2));
r = []; c = [];
for ii=1:size(proposals,2)
    box = proposals(:,ii);
    boxVMap = vGCMap(box(2):box(4),box(1):box(3));
    tempMin =  min(min(boxVMap));
    [rr,cc] = find(boxVMap==tempMin); % <------
    ridx = randi(numel(rr));
    r(ii) = rr(ridx);
    c(ii) = cc(ridx);
end

r = (r - ones(size(r)))';
c = (c - ones(size(c)))';
proposals = proposals-ones(size(proposals)); % C++ indexing for mex
nNeighbors = getGridLLEMatrixFeatures_rp_tol(normGCmap, r, c, proposals, 10, size(normGCmap,3), tol);

% % nNeighbors = getGridLLEMatrixFeatures_Global_tol(fMap, r, c, proposals, 50, size(fMap,3), tol);
% % fid = fopen('/home/saurabh/WORK/CODES/intrinsic_texture-master_focalStack/X.yml', 'r') ;              % Open source file.
% % fgetl(fid) ;                                  % Read/discard line.
% % fgetl(fid) ;                                  % Read/discard line.
% % buffer = fread(fid, Inf) ;                    % Read rest of the file.
% % fclose(fid);
% % fid = fopen('/home/saurabh/WORK/CODES/intrinsic_texture-master_focalStack/X.yml', 'w')  ;   % Open destination file.
% % fwrite(fid, buffer) ;                         % Save to file.
% % fclose(fid) ;
% % Y = ReadYaml('/home/saurabh/WORK/CODES/intrinsic_texture-master_focalStack/X.yml');
% % Ydata = reshape(Y.data,[Y.rows Y.cols]);
% % Ydata = cell2mat(Ydata);


% local computation
varianceOfFMap=zeros(1,Nn);
for ii=1:size(fMapXY,3)
    varianceOfFMap = varianceOfFMap + var(im2col(padarray(fMapXY(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
end
vMap = reshape(varianceOfFMap, [hn wn]);
mFMapXY_p = permute(vMap, [2 1 3]);
nNeighbors2_s = getGridLLEMatrixFeatures_Local_tol(fMapXY, mFMapXY_p, 50, size(fMapXY,3), g_size, tol); % Local neighbourhoods constraints

%% Dense Constraints
C = getChrom(S);
thres = 0.001;
nthres = 0.001;
wr = ones(hn, wn);
ws = 1.0;
[consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);

% Compute propagation weights (matting Laplacian, surface normal)
disp('computing laplacian matrix.');
L_S = getLaplacian1(S, zeros(hn, wn), 0.1^5);
% nweightMap = getFeatureConstraintMatrix(fMap, sig_n, size(fMap,3));
nweightMap = getFeatureConstraintMatrix(fMapXY, sig_n, size(fMapXY,3));

% Compute local reflectance constraint (continuous similarity weight)
[consVecCont_s, weightMap_s] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);

%% Optimization
spI = speye(Nn, Nn);
mk = zeros(Nn, 1);
mk(nNeighbors(:, 1)) = 1;
mask = spdiags(mk, 0, Nn, Nn); % Subsampling mask for local LLE
mk = zeros(Nn, 1);
mk(nNeighbors2_s(:, 1)) = 1;
mask2 = spdiags(mk, 0, Nn, Nn); % Subsampling mask for non-local LLE

A = 4 * WRC + 0.1 * mask * (spI - LLELOCAL) + 0.05 * mask2 * (spI - LLEGLOBAL) + 1 * L_S + 0.025 * WSC;
b = 4 * consVecCont_s;

disp('Optimizing the system...');
newS = pcg(A, b, 1e-3, 2000, [], []);

%% Visualization and Saving Results
res_s = reshape(exp(newS), [hn wn])/2;
res_r = I ./ repmat(res_s, [1 1 3]) /2;

resPath = fullfile(resDir,[imName '_' ]);
imwrite([I res_r repmat(res_s,[1 1 3])], [resPath 'all.png']);
imwrite(res_r,[resPath 'R.png']);
imwrite(res_s,[resPath 'S.png']);

end
