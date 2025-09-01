#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include "core/pose.h"


struct Frame {
  int id = -1;
  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<int> kp_to_landmark;  // landmark id for each keypoint, -1 if none
  Pose T_cw; // camera pose wrt world

    // association to landmarks by global index
  std::vector<int> landmark_ids;        // size N, -1 if none
  // Frame() {}
  void resizeAssociation() { kp_to_landmark.assign(keypoints.size(), -1); }
};
