#include "Stag.h"
#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cout << "Incorrect number of arguments.\n"
      "Usage: testrun 'path to image' 'class'\n";
  }
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  Stag stag(std::stoi(argv[2]), 7, true);

  

  stag.detectMarkers(image);
  stag.logResults("");
  
  return 0;
}