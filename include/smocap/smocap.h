#ifndef SMOCAP_H
#define SMOCAP_H

#include <opencv2/opencv.hpp>

// Example in: /home/poine/src/opencv_ros/tmp/opencv/samples/cpp/detect_blob.cpp

namespace smocap {

  class SMoCap {

  public:
    SMoCap();
    void detectMarkers(const cv::Mat img);

  private:
    cv::SimpleBlobDetector::Params detector_params_;
    cv::Ptr<cv::SimpleBlobDetector> detector_;

    
  };

}


#endif // SMOCAP_H
