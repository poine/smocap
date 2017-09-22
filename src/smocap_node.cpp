#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

class SmocapNode {
public:
  SmocapNode();
private:
  void GrabImage(const sensor_msgs::ImageConstPtr& msg);
  ros::NodeHandle nh_ ;
};

SmocapNode::SmocapNode() {
  std::cerr << "whello " << std::endl;
  nh_ = ros::NodeHandle();
  ros::Subscriber _img_sub = nh_.subscribe("/smocap/camera/image_raw", 1, &SmocapNode::GrabImage, this);
}
  
void SmocapNode::GrabImage(const sensor_msgs::ImageConstPtr& msg) {
  ROS_INFO("Blaaa %f", msg->header.stamp.toSec());
  std::cerr << "hello " << std::endl;
}
  
int main(int argc, char** argv){
  ros::init(argc, argv, "smocap_node");
  const SmocapNode& smn = SmocapNode();
  try{
    ros::spin();
  }catch(std::runtime_error& e){
    ROS_ERROR("smocap exception: %s", e.what());
    return -1;
  }

  return 0;
}
