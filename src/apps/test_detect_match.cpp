#include <iostream>
#include <opencv2/opencv.hpp>
#include "frontend/feature_detector.h"
#include "frontend/feature_matcher.h"

int main(int argc, char** argv){
  if (argc < 3) {
    std::cerr << "Usage: ./test_detect_match <img1> <img2>\n";
    return 1;
  }
  cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Failed to load images\n"; return 1;
  }

  Frame f1, f2; f1.image = img1; f2.image = img2;

  FeatureDetector detector({});
  detector.detectAndDescribe(f1);
  detector.detectAndDescribe(f2);
  std::cout << "kpts1=" << f1.keypoints.size()
            << " kpts2=" << f2.keypoints.size() << "\n";

  FeatureMatcher matcher({});
  auto matches = matcher.match(f1, f2);
  std::cout << "matches=" << matches.size() << "\n";

  // quick viz
  std::vector<cv::DMatch> cvm; cvm.reserve(matches.size());
  for (auto& m : matches) cvm.emplace_back(m.i1, m.i2, m.distance);
  cv::Mat vis;
  cv::drawMatches(img1, f1.keypoints, img2, f2.keypoints, cvm, vis);
  cv::imshow("matches", vis);
  // Save the image as "output.png"
  bool success = cv::imwrite("output_matching.png", vis); 
  cv::waitKey(0);
  return 0;
}
