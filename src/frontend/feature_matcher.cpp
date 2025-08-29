#include <opencv2/features2d.hpp>
#include "frontend/feature_matcher.h"

FeatureMatcher::FeatureMatcher(const MatchOptions& opt) : opt_(opt) {}

std::vector<Match> FeatureMatcher::match(const Frame& f1, const Frame& f2) const {
  std::vector<Match> out;
  if (f1.descriptors.empty() || f2.descriptors.empty()) return out;

  cv::BFMatcher matcher(cv::NORM_HAMMING, false);

  // forward KNN (for ratio)
  std::vector<std::vector<cv::DMatch>> knn12;
  matcher.knnMatch(f1.descriptors, f2.descriptors, knn12, 2);

  // optional reverse best-match table for cross-check
  std::vector<int> best21;
  if (opt_.cross_check) {
    std::vector<std::vector<cv::DMatch>> knn21;
    matcher.knnMatch(f2.descriptors, f1.descriptors, knn21, 1);
    best21.assign((size_t)f2.descriptors.rows, -1);
    for (size_t j = 0; j < knn21.size(); ++j) {
      if (!knn21[j].empty()) best21[j] = knn21[j][0].trainIdx;
    }
  }

  out.reserve(knn12.size());
  for (size_t i = 0; i < knn12.size(); ++i) {
    if (knn12[i].size() < 2) continue;
    const auto& m0 = knn12[i][0];
    const auto& m1 = knn12[i][1];
    if (m0.distance < opt_.ratio * m1.distance) {
      if (opt_.cross_check) {
        if (best21[m0.trainIdx] == (int)m0.queryIdx) {
          out.push_back({(int)m0.queryIdx, (int)m0.trainIdx, m0.distance});
        }
      } else {
        out.push_back({(int)m0.queryIdx, (int)m0.trainIdx, m0.distance});
      }
    }
    if ((int)out.size() >= opt_.max_matches) break;
  }
  return out;
}
