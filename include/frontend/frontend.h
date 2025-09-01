#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "types/frame.h"
#include "types/landmark.h"
#include "core/camera.h"
#include "frontend/feature_detector.h"
#include "frontend/feature_matcher.h"
#include "frontend/initializer.h"
#include "frontend/triangulator.h"
#include "frontend/pose_estimator.h"

class Frontend {
 public:
  explicit Frontend(const PinholeCamera& cam);

  // initialize with first two frames
  bool initialize(Frame& f1, Frame& f2);

  // track a new frame, returns true if successful pose estimation
  bool trackFrame(Frame& f);

  // access map
  const std::vector<Frame>& keyframes() const { return keyframes_; }
  const std::vector<Landmark>& landmarks() const { return landmarks_; }

 private:
  // helper: promote to keyframe, triangulate with last keyframe
  void insertKeyframe(Frame& f);

  // data
  PinholeCamera cam_;
  FeatureDetector detector_;
  FeatureMatcher matcher_;
  Initializer initializer_;
  Triangulator triangulator_;
  PoseEstimator pnp_;

  std::vector<Frame> keyframes_;
  std::vector<Landmark> landmarks_;
  int next_landmark_id_ = 0;
};
