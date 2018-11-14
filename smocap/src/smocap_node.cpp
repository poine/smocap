#include <chrono>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "smocap/smocap.h"

namespace smocap {
static const std::string OPENCV_WINDOW = "Image window";

  class SMocapNode
  {
  
  public:
    SMocapNode();
    ~SMocapNode();
    
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    
    SMoCap smocap_;

    ros::Time last_img_stamp_;
    ros::Duration img_interval_;
    double fps_;
    
    void imageCb(const sensor_msgs::ImageConstPtr& msg);
    
  };
  
  /*******************************************************************************
   *
   *
   *******************************************************************************/
  SMocapNode::SMocapNode()
    : nh_(),
      it_(nh_)
  {
    image_sub_ = it_.subscribe("/camera/image_raw", 1,
			       &SMocapNode::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    
    //cv::namedWindow(OPENCV_WINDOW);
  }
  /*******************************************************************************
   *
   *
   *******************************************************************************/
  SMocapNode::~SMocapNode()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }
  /*******************************************************************************
   *
   *
   *******************************************************************************/
  void SMocapNode::imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    //cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImageConstPtr cv_ptr2;
    try
      {
	//cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	cv_ptr2 = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
      }
    catch (cv_bridge::Exception& e)
      {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }

    img_interval_ = msg->header.stamp - last_img_stamp_;
    last_img_stamp_ = msg->header.stamp;
    fps_ = 1./img_interval_.toSec();
    std::cerr << fps_ << std::endl;
    //smocap_.detectMarkers(cv_ptr->image);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    smocap_.detectMarkers(cv_ptr2->image);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
    std::cerr << duration << std::endl;
    // Draw an example circle on the video stream
    //if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //  cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
    
    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::waitKey(3);
    
    // Output modified video stream
    //image_pub_.publish(cv_ptr->toImageMsg());
    image_pub_.publish(cv_ptr2->toImageMsg());
  }
  
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "smocap_node");
  smocap::SMocapNode ic;
  ros::spin();
  return 0;
}
