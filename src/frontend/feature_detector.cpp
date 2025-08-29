#include <opencv2/features2d.hpp>
#include "frontend/feature_detector.h"

FeatureDetector::FeatureDetector(const FeatureDetectorCfg& cfg) : cfg_(cfg) {}

void FeatureDetector::detectAndDescribe(Frame& f) const {
  CV_Assert(!f.image.empty());
  CV_Assert(f.image.channels() == 1); // grayscale

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        cfg_.n_features,
        cfg_.scale_factor,
        cfg_.n_levels,
        cfg_.edge_threshold,
        0, // firstLevel
        2, // WTA_K (ORB-SLAM uses 2)
        cv::ORB::HARRIS_SCORE,
        cfg_.patch_size,
        cfg_.fast_threshold);

        f.keypoints.clear();
        orb->detectAndCompute(f.image, cv::noArray(), f.keypoints, f.descriptors);
        f.landmark_ids.assign(f.keypoints.size(), -1);


}