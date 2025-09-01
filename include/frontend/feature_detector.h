#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include "types/frame.h"

// ORB-style detector/descriptor, ORB-SLAM-style defaults.
struct FeatureDetectorCfg {
  int   n_features   = 1500;   // total per frame
  int   n_levels     = 10;      // pyramid levels
  float scale_factor = 1.2f;   // pyramid scale
  int   edge_threshold = 31;   // ORB default
  int   patch_size     = 31;   // ORB default
  int   fast_threshold = 20;   // FAST threshold
};

class FeatureDetector {
 public:
  explicit FeatureDetector(const FeatureDetectorCfg& cfg);
  // Detect keypoints and compute descriptors in-place on the Frame (expects grayscale)
  void detectAndDescribe(Frame& f) const;
  const FeatureDetectorCfg& config() const { return cfg_; }
 private:
  FeatureDetectorCfg cfg_;
};
