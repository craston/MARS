#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;



// compilation with opencv4
// OPENCV=<path_where_opencv_is_installed>
// g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I${OPENCV}include/opencv4/ -L${OPENCV}lib64 -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm


void writeFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, const char *filename)
{
  /*
    save a flow into the standard .flo binary format
  */ 
  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
    {
      printf("Error while opening %s\n",filename);
      exit(1);
    }
  float help=202021.25;
  int dummy;
  dummy = fwrite(&help,sizeof(float),1,stream);
  int aXSize = flowx.cols, aYSize = flowx.rows;
  fwrite(&aXSize,sizeof(int),1,stream);
  fwrite(&aYSize,sizeof(int),1,stream);
  int y,x;
  float data;
  for (y = 0; y < aYSize ; y++)
    for (x = 0; x < aXSize ; x++)
      {
        Point2f u(flowx(y, x), flowy(y, x));
        data = u.x;
        fwrite(&data,sizeof(float),1,stream);
        data = u.y;
        fwrite(&data,sizeof(float),1,stream);
      }
  fclose(stream);
}



int main(int argc, const char* argv[])
{
    char infileformat[2000], outfileformat[2000], im1name[2000], im2name[2000], outname[2000];
    int length;
    if (argc != 4)
    {
        cerr << "Usage : " << argv[0] << " <infileformat> <nframes> <outfileformat>" << endl;
        exit(1);
    }
    length = atoi(argv[2]);
    
    for( int i=1 ; i<length ; ++i){ 

      sprintf(im1name, argv[1], i);
      sprintf(im2name, argv[1], i+1);
      sprintf(outname, argv[3], i);
      Mat frame0 = imread(im1name, IMREAD_GRAYSCALE);
      Mat frame1 = imread(im2name, IMREAD_GRAYSCALE);

      if (frame0.empty())
      {
          cerr << "Can't open image ["  << im1name << "]" << endl;
          return -1;
      }
      if (frame1.empty())
      {
          cerr << "Can't open image ["  << im2name << "]" << endl;
          return -1;
      }

      if (frame1.size() != frame0.size())
      {
          cerr << "Images should be of equal sizes" << endl;
          return -1;
      }

      GpuMat d_frame0(frame0);
      GpuMat d_frame1(frame1);

      GpuMat d_flow(frame0.size(), CV_32FC2);

      Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

      tvl1->calc(d_frame0, d_frame1, d_flow);

      GpuMat planes[2];
      cuda::split(d_flow, planes);

      Mat flowx(planes[0]);
      Mat flowy(planes[1]);
      
      writeFlow(flowx, flowy, outname);
    
    }
    return 0;
}
