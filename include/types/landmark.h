#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>

struct Landmark {
  int id = -1;
  Eigen::Vector3d Xw = Eigen::Vector3d::Zero();
  int seen = 0;      // total observations
  int inliers = 0;   // times it was an inlier
  bool bad = false;  // culled?
  cv::Mat descriptor; // representative descriptor (1x32 CV_8U for ORB)
};
