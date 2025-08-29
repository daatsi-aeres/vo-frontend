#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "types/frame.h"
#include "core/camera.h"

struct InitOptions {
  double ransac_thresh_px = 1.0;
  double ransac_conf = 0.999;
};

struct InitResult {
  bool ok = false;
  Eigen::Matrix3d R;          // relative pose (1->2)
  Eigen::Vector3d t;
  std::vector<int> inlier_idx;// indices into the matched pair array
};

class Initializer {
 public:
  explicit Initializer(const InitOptions& opt);
  // inputs: keypoint pairs in pixel coordinates (from your matcher), and K
  InitResult estimateRelativePose(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  const PinholeCamera& cam) const;
 private:
  InitOptions opt_;
};
