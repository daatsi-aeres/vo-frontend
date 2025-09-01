#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "core/camera.h"

struct PnPOptions {
  double ransac_thresh_px = 3.0;
  double conf = 0.999;
  int    max_iters = 1000;
  bool   use_refine = false; // (optional) refine with one Gauss-Newton step
};

struct PnPResult {
  bool ok = false;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  std::vector<int> inliers; // indices into the input correspondences
  double mean_reproj_err = -1.0;
};

class PoseEstimator {
 public:
  explicit PoseEstimator(const PnPOptions& opt);
  // Xw: 3D points (world frame). uv: pixels in current frame.
  PnPResult solve(const std::vector<Eigen::Vector3d>& Xw,
                  const std::vector<cv::Point2f>& uv,
                  const PinholeCamera& cam) const;

 private:
  PnPOptions opt_;
};
