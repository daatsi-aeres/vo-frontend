#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "types/frame.h"
#include "types/landmark.h"
#include "core/camera.h"
#include "frontend/feature_detector.h"
#include "frontend/feature_matcher.h"
#include "frontend/initializer.h"
#include "frontend/triangulator.h"
#include "frontend/pose_estimator.h"



#ifdef USE_PANGOLIN
#include "visualization/pangolin_visualizer.h"   // ✅ Pangolin visualizer
#endif

class Frontend {
public:
    explicit Frontend(const PinholeCamera& cam);

    // --- main API ---
    bool initialize(Frame& f1, Frame& f2);
    bool trackFrame(Frame& f);
    void insertKeyframe(Frame& f);
    void reprojectAndMatch(Frame& f, const Pose& pose);

    // --- helper ---
    static int descriptorDistance(const cv::Mat& d1, const cv::Mat& d2);

    #ifdef USE_PANGOLIN
        PangolinVisualizer& visualizer() { return vis_; } 
        PangolinVisualizer vis_;
    #endif

private:
    // --- configuration ---
    PinholeCamera cam_;

    // --- core modules ---
    FeatureDetector detector_;
    FeatureMatcher matcher_;
    Initializer initializer_;
    Triangulator triangulator_;
    PoseEstimator pnp_;

    // --- state ---
    std::vector<Frame> keyframes_;
    std::vector<Landmark> landmarks_;
    std::vector<Pose> trajectory_;
    int next_landmark_id_ = 0;


};

