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
  // const int window_size = 3;

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
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  ////////////////////
  // Reconstruction //
  ////////////////////

  // Naive disparity image
  cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  //Get the time before calling the function
  auto start = high_resolution_clock::now();

  StereoEstimation_Naive(
    window_size, dmin, height, width,
    image1, image2,
    naive_disparities);

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
    height, width, naive_disparities,
    window_size, dmin, baseline, focal_length);


  // save / display images for naive
  std::stringstream out1;
  out1 << output_file << "_naive.png";
  cv::imwrite(out1.str(), naive_disparities);

  cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
  cv::imshow("Naive", naive_disparities);

  cv::waitKey(0);

  // saving the time taken to get the results in a text file
  std::ofstream txt_file;
  // File Open
  txt_file.open(output_file +".txt");

  // Write to the file
  txt_file << "Window_size = "<< window_size << std::endl;
  txt_file << "Time Taken = "<< seconds << std::endl;

  // File Close
  txt_file.close();

  return 0;
}

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
  int half_window_size = window_size / 2;
  #pragma omp parallel for
  for (int i = half_window_size; i < height - half_window_size; ++i) {

    std::cout
      << "Calculating disparities for the naive approach... "
      << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
      << std::flush;

    for (int j = half_window_size; j < width - half_window_size; ++j) {
      int min_ssd = INT_MAX;
      int disparity = 0;

      for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
        int ssd = 0;

        // TODO: sum up matching cost (ssd) in a window
        for (int x=-half_window_size;x<=half_window_size;++x){
          for (int y=-half_window_size;y<=half_window_size;++y)
          {
            ssd=ssd+pow((int)image1.at<uchar>(i+x,j+y)-(int)image2.at<uchar>(i+x,j+y+d),2);
          }
        }
        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i , j) = std::abs(disparity);

    }
  }
  std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
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