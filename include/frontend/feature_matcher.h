#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "types/frame.h"

// Simple Hamming matcher with ratio test and optional cross-check.
struct Match { int i1 = -1; int i2 = -1; float distance = 0.f; };

struct MatchOptions {
  double ratio = 0.9;          // Lowe's ratio test
  bool   cross_check = false;  // mutual best
  int    max_matches = 50;   // safety cap
};

class FeatureMatcher {
 public:
  explicit FeatureMatcher(const MatchOptions& opt);
  // Returns index pairs into f1.keypoints/f2.keypoints
  std::vector<Match> match(const Frame& f1, const Frame& f2) const;
  const MatchOptions& options() const { return opt_; }
 private:
  MatchOptions opt_;
};
