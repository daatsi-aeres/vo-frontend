// frame.h
struct Feature2D {
Eigen::Vector2d uv; // pixel coordinates
Eigen::Vector2d xn; // normalized coords (optional cache)
cv::Mat descriptor; // 1 x 32 (ORB)
int landmark_id = -1; // linked 3D point
};


struct Frame {
int id;
double timestamp;
cv::Mat image;
Pose T_cw; // pose estimate (world->camera)
std::vector<cv::KeyPoint> kps;
cv::Mat descriptors; // N x 32
std::vector<int> lm_ids; // size N, -1 if none
};