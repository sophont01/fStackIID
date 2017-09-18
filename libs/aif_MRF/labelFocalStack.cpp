//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////
//
//  Optimization problem:
//  is a set of sites (pixels) of width 10 and hight 5. Thus number of pixels is 50
//  grid neighborhood: each pixel has its left, right, up, and bottom pixels as neighbors
//  7 labels
//  Data costs: D(pixel,label) = 0 if pixel < 25 and label = 0
//            : D(pixel,label) = 10 if pixel < 25 and label is not  0
//            : D(pixel,label) = 0 if pixel >= 25 and label = 5
//            : D(pixel,label) = 10 if pixel >= 25 and label is not  5
// Smoothness costs: V(p1,p2,l1,l2) = min( (l1-l2)*(l1-l2) , 4 )
// Below in the main program, we illustrate different ways of setting data and smoothness costs
// that our interface allow and solve this optimizaiton problem

// For most of the examples, we use no spatially varying pixel dependent terms. 
// For some examples, to demonstrate spatially varying terms we use
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GCoptimization.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;

struct ForDataFn{
	int numLab;
	int *data;
};


int smoothFn(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}

int dataFn(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *) data;
	int numLab = myData->numLab;
	
	return( myData->data[p*numLab+l] );
}



////////////////////////////////////////////////////////////////////////////////
// smoothness and data costs are set up one by one, individually
// grid neighborhood structure is assumed
//
void GridGraph_Individually(int width,int height,int num_pixels,int num_labels, Mat *gxgx, Mat *gygy, Mat *gxgy, Mat *res, Mat *lab)
{

	int *result = new int[num_pixels];   // stores result of optimization

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually
		for ( int i=0; i<num_pixels; i++ )
		{
			int row = i/width;
			int col = i%width;

			for (int l=0; l<num_labels; l++ )
			{

				float gxx = gxgx[l].at<float>(row,col); 
		 		float gyy = gygy[l].at<float>(row,col);
		 		float gxy = gxgy[l].at<float>(row,col);
				
				float trace = gxx+gyy;
				float determinant = gxx*gyy - gxy*gxy;
				
				double cost = -0.05*(trace);
				
				//cout<<gxx<<endl;
				//cout<<gyy<<endl;
				//cout<<cost<<endl;

				gc->setDataCost(i,l,cost);
			}
		}
		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ ){
				if(l1==l2) gc->setSmoothCost(l1,l2,0);
				else gc->setSmoothCost(l1,l2,350); //Potts multi-label model
			}


		printf("Before optimization energy is %lld\n",gc->compute_energy());
		gc->expansion(20);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("After optimization energy is %lld\n",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
		{	
			int row = i/width;
			int col = i%width;
			
			result[i] = gc->whatLabel(i);
			res->at<unsigned char>(row,col)=result[i]*(255/num_labels); //result*(255/num_labels)
			lab->at<unsigned char>(row,col)=result[i]; //result
		}

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
}
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	int numSlices = atoi(argv[1]);

	string folderName = argv[2]; 
	
	cv::Mat gc[numSlices], g[numSlices];
	cv::Mat gxgx[numSlices], gygy[numSlices], gxgy[numSlices];
	for(int i=0;i<numSlices;i++)
	{
		//string Prefix = "../data/"+folderName+"/f"; //"../1_OriginalAligned/";
		string Prefix = folderName+"/f"; //"../1_OriginalAligned/";
		stringstream NameStream;
		NameStream<<(i+1);
		string Postfix = ".png";
		string Name = Prefix + NameStream.str() + Postfix;
		cout<<Name<<endl;

		gc[i] = cv::imread(Name, CV_LOAD_IMAGE_COLOR);

		cvtColor(gc[i],g[i],COLOR_BGR2GRAY);

		Size kernelSize = Size(5,5);
		cv::GaussianBlur(g[i],g[i],kernelSize,0.6,0.6,BORDER_REPLICATE);	

		cv::Mat templateG = g[i].clone();
		//gxgx[i] = g[i].clone();
		//gygy[i] = g[i].clone();
		//gxgy[i] = g[i].clone();
		
		templateG.convertTo(gxgx[i], CV_32F);
		templateG.convertTo(gygy[i], CV_32F);
		templateG.convertTo(gxgy[i], CV_32F);
		
		//cvNamedWindow( "Image", CV_WINDOW_AUTOSIZE );
		//imshow("Image", gxgx[i]);
		//cvWaitKey(0);
	}

	
	Size s = g[0].size();
	int width = s.width;
	int height = s.height;

	cv::Mat Result = g[0].clone();
	cv::Mat Labels = g[0].clone();

	int startX=1,  endX=width-1, startY=1, endY=height-1;

	for( int i=0;i<numSlices;i++)
	{
		
		cv::Mat gx,gy;
		//Mat gx = g[i].clone();
		//Mat gy = g[i].clone();
		cv::Mat templateG = g[i].clone();
		templateG.convertTo(gx, CV_32F);
		templateG.convertTo(gy, CV_32F);
		Sobel(g[i], gx, CV_32F, /*xOrder*/ 1, /*yOrder*/ 0, /*kernelSize*/ 3, BORDER_REPLICATE); 
		Sobel(g[i], gy, CV_32F, /*xOrder*/ 0, /*yOrder*/ 1, /*kernelSize*/ 3, BORDER_REPLICATE); 
		
		
		float gxx = gx.at<float>(100,100); 
		float gyy = gy.at<float>(100,100);
		cout<<gxx<<endl;
		cout<<gyy<<endl;
		cout<<gx.size()<<endl;

		gxgx[i] = gx.mul(gx);
		gygy[i] = gy.mul(gy);
		gxgy[i] = gx.mul(gy);
	
		gxx = gxgx[i].at<float>(100,100); 
		gyy = gygy[i].at<float>(100,100);
		cout<<gxx<<endl;
		cout<<gyy<<endl;
		cout<<endl;
		
		Size kernelSize = Size(5,5);
		cv::GaussianBlur(gxgx[i],gxgx[i],kernelSize,1.7,1.7,BORDER_REPLICATE);	
		cv::GaussianBlur(gygy[i],gygy[i],kernelSize,1.7,1.7,BORDER_REPLICATE);	
		cv::GaussianBlur(gxgy[i],gxgy[i],kernelSize,1.7,1.7,BORDER_REPLICATE);	
		
		
			
		//cvNamedWindow( "Image", CV_WINDOW_AUTOSIZE );
		//imshow("Image", gxgx[i]);
		//cvWaitKey(0);
	}
	
	
	// smoothness and data costs are set up one by one, individually
	GridGraph_Individually(width, height, width*height, numSlices, gxgx, gygy, gxgy, &Result, &Labels);

	//cv::imwrite("./LabelResult.jpg", Result);

	for(int i=0;i<width*height;i++)
	{	
		int row = i/width;
		int col = i%width;

		for(int l=0;l<numSlices;l++)
		{
			if(Labels.at<unsigned char>(row,col)!=l)
			{
				gc[l].at<Vec3b>(row,col)[0]=0;
				gc[l].at<Vec3b>(row,col)[1]=0;
				gc[l].at<Vec3b>(row,col)[2]=0;
			}
		}
	}
	/*for(int i=0;i<numSlices;i++)
	{
		string Prefix = "./fTexturesColor/";
		stringstream NameStream;
		NameStream<<(i+1);
		string Postfix = ".png";
		string Name = Prefix + NameStream.str() + Postfix;
		cout<<Name<<endl;

		imwrite(Name, gc[i]);
	}*/

	printf("\n  Finished %ld (%ld) clock per sec %ld\n",clock()/CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);
	

	// Write AiF image
	cv::Mat Fa[numSlices];
	Fa[0] = gc[0];
	for(int k=1;k<numSlices;k++)
	{
		add(Fa[0],gc[k],Fa[0]);
	}
	cv::Mat AIF=Fa[0];
	String targetString = folderName+"/aif.png";
	cv::imwrite(targetString,AIF);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////

