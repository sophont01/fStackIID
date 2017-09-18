projPath = mfilename('fullpath');
projPath = fileparts(projPath);
fStackDataPath = fullfile(projPath, ['./DATA/fStack_1/']);
fStackResPath = fullfile(projPath, ['./RESULTS/']);

fStackIID(fStackDataPath, fStackResPath);