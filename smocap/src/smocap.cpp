#include "smocap/smocap.h"

#include <vector>

namespace smocap {

  SMoCap::SMoCap() {
    detector_params_.thresholdStep = 10;
    detector_params_.minThreshold = 2;
    detector_params_.maxThreshold = 256;
    detector_params_.minRepeatability = 2;
    detector_params_.minDistBetweenBlobs = 10;
    detector_params_.filterByColor = false;
    detector_params_.blobColor = 0;
    detector_params_.filterByArea = true;
    detector_params_.minArea = 24;
    detector_params_.maxArea = 500;
    detector_params_.filterByCircularity = false;
    detector_params_.minCircularity = 0.9f;
    detector_params_.maxCircularity = (float)1e37;
    detector_params_.filterByInertia = false;
    detector_params_.minInertiaRatio = 0.1f;
    detector_params_.maxInertiaRatio = (float)1e37;
    detector_params_.filterByConvexity = false;
    detector_params_.minConvexity = 0.95f;
    detector_params_.maxConvexity = (float)1e37;

    detector_ = cv::SimpleBlobDetector::create(detector_params_);

    
  }


  void SMoCap::detectMarkers(const cv::Mat img) {
    std::vector<cv::KeyPoint>  keyImg;
    detector_->detect(img, keyImg, cv::Mat());
    std::cerr << keyImg.size() << std::endl;
  }
  
  
  
}
