function[normFeatures]= normalizeGCFeatures(features)
normFeatures = zeros(size(features));

%% Normalizing features for each section spearately

%      01 - 03: mean rgb
normFeatures(1:3) = features(1:3)./norm(features(1:3));
%      04 - 06: hsv conversion
normFeatures(4:6) = features(4:6)./norm(features(4:6));
%      07 - 11: hue histogram
normFeatures(7:11) = features(7:11)./norm(features(7:11));
%      12 - 14: sat histogram
normFeatures(12:14) = features(12:14)./norm(features(12:14));
%      15 - 29: mean texture response
normFeatures(15:29) = features(15:29)./norm(features(15:29));
%      30 - 44: texture response histogram
normFeatures(30:44) = features(30:44)./norm(features(30:44));
% %      45 - 46: mean x-y
% normFeatures(45:46) = features(45:46);
% %      47 - 48: 10th, 90th perc. x
% normFeatures(47:48) = features(47:48);
% %      49 - 50: 10th, 90th perc. y
% normFeatures(49:50) = features(49:50);
% %      51 - 51: h / w 
% normFeatures(51) = features(51);
% %      52 - 52: area
% normFeatures(52) = features(52);

normFeatures=normFeatures';

end