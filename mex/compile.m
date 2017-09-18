% PROVIDE appropriate paths to opencv and ANN libararies before compiling 


% REQUIRE opencv
mex -v -I"/usr/local/include/opencv/" -L"/usr/local/lib/" -l"opencv_core" getConstraintsMatrix.cpp mexBase.cpp
mex -v -I"/usr/local/include/opencv/" -L"/usr/local/lib/" -l"opencv_core" getFeatureConstraintMatrix.cpp mexBase.cpp
mex -v -I"/usr/local/include/opencv/" -L"/usr/local/lib/" -l"opencv_core" getContinuousConstraintMatrix.cpp mexBase.cpp


%% GRID Features ( REQUIRE ANN+OPENCV)
mex -I"/media/saurabh/String/SOFTWARES/ANN/ann_1.1.2/include/ANN/" -L"/media/saurabh/String/SOFTWARES/ANN/ann_1.1.2/lib/" -l"ANN" -I"/usr/local/include/opencv/" -L"/usr/local/lib/" -l"opencv_core" getGridLLEMatrixFeatures_rp_tol.cpp mexBase.cpp LLE_fStack_tol.cpp
mex -I"/media/saurabh/String/SOFTWARES/ANN/ann_1.1.2/include/ANN/" -L"/media/saurabh/String/SOFTWARES/ANN/ann_1.1.2/lib/" -l"ANN" -I"/usr/local/include/opencv/" -L"/usr/local/lib/" -l"opencv_core" getGridLLEMatrixFeatures_Local_tol.cpp mexBase.cpp LLE_fStack_tol.cpp

