#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"
#include <chrono>
using namespace std::chrono;

int main(int argc, char** argv) {

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 160;

  // stereo estimation parameters
  const int dmin = 200;
  // const int window_size =1;
  // int lambda =1;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE" << std::endl;
    return 1;
  }

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  const std::string output_file = argv[3];
  int window_size =std::atoi(argv[4]);
  int lambda = std::atoi(argv[5]);

  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size = " << window_size << std::endl;
  std::cout << "lambda = " << lambda << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  ////////////////////
  // Reconstruction //
  ////////////////////

  // Dynamic Programming disparity image  
  cv::Mat dp_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  //Get the time before calling the function
  auto start = high_resolution_clock::now();

  StereoEstimation_DP(
    window_size,height,width,
    lambda,image1, image2, dp_disparities);

  //Get time after the function is implemented
  auto stop = high_resolution_clock::now();
 
  // Calculate the time taken in minutes and seconds
  auto duration = duration_cast<seconds>(stop - start);
  int seconds = duration.count();
  int minutes = seconds / 60;
  std::cout << "Time taken = " << minutes<< "m" <<" "<< int(seconds%60) << "s"<<std::endl;
 
  ////////////
  // Output //
  ////////////

  // reconstruction

  Disparity2PointCloud(
    output_file,
    height, width,dp_disparities ,
    window_size, dmin, baseline, focal_length);


  // save and display images for Dynamic programming
  std::stringstream out2;
  out2 << output_file << "_dynamic.png";
  cv::imwrite(out2.str(), dp_disparities);

  cv::namedWindow("DP", cv::WINDOW_AUTOSIZE);
  cv::imshow("DP", dp_disparities);

  cv::waitKey(0);
 
  // saving the time taken to get the results in a text file
  std::ofstream txt_file;
  // File Open
  txt_file.open(output_file +".txt");

  // Write to the file
  txt_file << "Window_size = "<< window_size << std::endl;
  txt_file << "Lambda = "<< lambda << std::endl;
  txt_file << "Time Taken = "<< seconds << std::endl;

  // File Close
  txt_file.close();

  return 0;
}

void StereoEstimation_DP(
  const int& window_size,
  int height,
  int width,
  int lambda,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities) 
{
  int half_window_size = window_size / 2;
  //  for each row(scanline)
  for (int y_0= half_window_size; y_0< height - half_window_size; ++y_0){
    std::cout
    << "Calculating disparities for the DP approach...  "
    << std::ceil(((y_0 - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
    << std::flush;

    cv::Mat dissim =cv::Mat::zeros(width,width,CV_32FC1);
    //  dissimalirty (i,j) for each (i,j)
    #pragma omp parallel for
    for (int i = half_window_size ; i < width - half_window_size; ++i) { //left image
      for (int j = half_window_size ; j < width - half_window_size; ++j) { //right image
        // TODO: sum up matching cost (ssd) in a window
        float sum = 0;
        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v){
            float i1 = static_cast<float>(image1.at<uchar>( y_0+ v, i + u));
            float i2 =static_cast<float>( image2.at<uchar>(y_0 + v, j + u));
            sum+=std::abs(i1-i2);} //SAD works nice with 1 window size
        }
        dissim.at<float>(i,j)=sum;
      }
    }
    cv::Mat C = cv::Mat::zeros(width, width, CV_32FC1);
    cv::Mat M = cv::Mat::zeros(width, width, CV_8UC1); // match 0, left_occ 1, right_occ 2
    
    //Put values of the first rows and columns in C matrix
    for (int col=0;col<width;++col){
      C.at<float>(0,col)=lambda*col;
    }
    for (int row=0;row<width;++row){
      C.at<float>(row,0)=lambda*row;
    }

    for (int i = half_window_size ; i < width-half_window_size ; ++i) { 
      for (int j = half_window_size ; j < width -half_window_size; ++j) {
        C.at<float>(i,j)=std::min({C.at<float>(i-1,j-1)+dissim.at<float>(i, j),C.at<float>(i-1,j)+lambda,C.at<float>(i,j-1)+lambda});

        if (C.at<float>(i,j) == C.at<float>(i-1,j)+lambda){ //left_occ
          M.at<uchar>(i,j)=1;
        }
        else if (C.at<float>(i,j) == C.at<float>(i,j-1)+lambda){//right_occ
          M.at<uchar>(i,j)=2;
        } 
      }
    }

    // Sink to Source
    int left_ind = width - half_window_size;
    int right_ind = width - half_window_size;
    int  col_loop= width - half_window_size;
    while (left_ind >0 && right_ind > 0) {
      //match
      if (M.at<uchar>(left_ind, right_ind) == 0) {
        dp_disparities.at<uchar>(y_0, col_loop) = left_ind - right_ind;
        left_ind--;
        right_ind--;
        col_loop--;
      }
      if (M.at<uchar>(left_ind, right_ind ) == 1) {
        left_ind--;  // left occlusion
      }
      if (M.at<uchar>(left_ind , right_ind) == 2) {
        right_ind--;  // right occlusion
      }
    }  
  }
  std::cout << "Calculating disparities for the Dynamic programming approach... Done.\r" << std::flush;
  std::cout << std::endl;
}

  
void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());

  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      const double u1 = j - width/2.;    
      const double v1 = i - height/2.;
      const double v2 = v1;
      double d = static_cast<double>(disparities.at<uchar>(i,j))+ dmin;
      const double u2 = u1+d;
  
      // TODO
      const double Z = (baseline*focal_length)/(d);
      const double X = -((baseline*(u1+u2))/(2*d));
      const double Y = (baseline*(v1))/d;

      outfile << X << " " << Y << " " << Z << std::endl;
    }
  }
  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
  std::cout << std::endl;
}