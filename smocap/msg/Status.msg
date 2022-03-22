#
# Smocap Status
#
Header header
# camera system
float32[] fps
uint32[]  skipped
float32[] ff_duration
float32[] roi_duration
# markers
bool[]    marker_localized
bool[]    marker_full_frame_observation
geometry_msgs/Pose[] marker_pose

