#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include "core/pose.h"

struct Frame {
  int id = -1;
  double timestamp = 0.0;
  cv::Mat image;               // grayscale
  Pose T_cw;                   // world->camera estimate

  // ORB features
  std::vector<cv::KeyPoint> keypoints; // pixel coords
  cv::Mat descriptors;                  // N x 32 (uchar)

  // association to landmarks by global index
  std::vector<int> landmark_ids;        // size N, -1 if none
};
