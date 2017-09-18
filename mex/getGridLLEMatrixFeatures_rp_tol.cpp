#include "mexBase.h"
#include "LLE.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int feature_dim = mxGetPr(prhs[5])[0];
	int img_h = mxGetM(prhs[0]); //imgHeight
	int img_w = mxGetN(prhs[0])/feature_dim; //imgWidth
	int N = img_h*img_w;
	int K = mxGetPr(prhs[4])[0]; //k-NN
	int dims[2] = {N, N};
	CvSparseMat* affinityMatrix = cvCreateSparseMat(2, dims, CV_32FC1);
	double *fMap = mxGetPr(prhs[0]); // feature Map
	double *rIdx = mxGetPr(prhs[1]); // min vMap rIdx
	double *cIdx = mxGetPr(prhs[2]); // min vMap cIdx
	double *gbox = mxGetPr(prhs[3]); //global proposals
	int gbox_num = mxGetM(prhs[2]);	
	mexPrintf("img_h=%d img_w=%d N=%d K=%d gbox_num=%d\n",img_h, img_w, N, K, gbox_num);

	int *c_pos = new int[gbox_num];
	int *r_pos = new int[gbox_num];
	cv::Mat1f X(gbox_num, feature_dim);
	
	double tol_d = mxGetPr(prhs[6])[0]; //LLE tolerance
	float tol;
	tol = (float)tol_d;
	mexPrintf("tol=%f \n",tol);

		for(int b=0; b<gbox_num; b++){ //gbox iteration
		int cmin = gbox[(0 + b*4)];
		int rmin = gbox[(1 + b*4)];	
		int cmax = gbox[(2 + b*4)];	
		int rmax = gbox[(3 + b*4)];
		c_pos[b] = int(rIdx[b]);
		r_pos[b] = int(cIdx[b]);
	//	mexPrintf("b=%d\n",b);
		for(int k=0; k<feature_dim; k++)
		{
			//X(b,k) = fMap[ k*img_h*img_w + r_pos[b]*img_h + c_pos[b] ];
			X(b,k) = fMap[ k*img_h*img_w + r_pos[b]*img_h + c_pos[b] ];
		}

	}
//	cv::FileStorage fileX("X.yml", cv::FileStorage::WRITE);
//	fileX << "X" << X;	
	//mexPrintf("%f %f %f %f %f %f %f %f %f %f \n",X(3,0),X(3,1),X(3,2),X(3,3),X(3,4),X(3,5),X(3,6),X(3,7),X(3,8),X(3,9));	

//	mexPrintf("Checkpoint 1\n");
	cv::Mat1f W(gbox_num, K);
	cv::Mat1i neighbors(gbox_num, K);
	LLE(X, W, neighbors, gbox_num, feature_dim, tol, K);
	
//	cv::FileStorage fileW("W.yml", cv::FileStorage::WRITE);
//	fileX << "W" << W;	
//	mexPrintf("Checkpoint 2\n");

	plhs[0] = mxCreateDoubleMatrix(gbox_num, K+1, mxREAL);
	double *neighborPixels = mxGetPr(plhs[0]);
	for(int n=0;n<gbox_num;n++) {
		int xp = r_pos[n];
		int yp = c_pos[n];
		int p = xp * img_h + yp;
		neighborPixels[n] = p + 1;
		for(int k=0;k<K;k++) {
			if(W(n, k) != 0) {
				int nIdx = neighbors(n, k);
				if(nIdx >= 0) {
					int xq = r_pos[nIdx];
					int yq = c_pos[nIdx];
					//int q = yq * img_w + xq;
					int q = xq * img_h + yq;
					((float*)cvPtr2D(affinityMatrix, p, q))[0] = W(n, k);
					neighborPixels[(k+1)*gbox_num + n] = q + 1;
				}
			}
		}
	}
	pushSparseMatrix(affinityMatrix, "LLEGLOBAL");
	cvReleaseSparseMat(&affinityMatrix);

//	mexPrintf("Checkpoint 3\n");
	delete [] c_pos;
	delete [] r_pos;
}
