#include <opencv2/calib3d.hpp>
#include "frontend/initializer.h"

Initializer::Initializer(const InitOptions& opt) : opt_(opt) {}

InitResult Initializer::estimateRelativePose(const std::vector<cv::Point2f>& pts1,
                                             const std::vector<cv::Point2f>& pts2,
                                             const PinholeCamera& cam) const {
  InitResult out;
  if (pts1.size() < 8 || pts2.size() < 8) return out;

  // camera center (principal point) + focal for OpenCV’s API
  double fx = cam.K(0,0), fy = cam.K(1,1);
  double f = (fx+fy)*0.5; // OK to use fx if square pixels; avg is fine
  cv::Point2d pp(cam.K(0,2), cam.K(1,2));

  cv::Mat inliersMask;
  cv::Mat E = cv::findEssentialMat(pts1, pts2, f, pp,
                                   cv::RANSAC, opt_.ransac_conf, opt_.ransac_thresh_px, inliersMask);
  if (E.empty()) return out;

  cv::Mat Rcv, tcv;
  int ninl = cv::recoverPose(E, pts1, pts2, Rcv, tcv, f, pp, inliersMask);
  if (ninl < 10) return out;

  // copy to Eigen
  out.R.setZero();
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) out.R(r,c) = Rcv.at<double>(r,c);
  out.t = Eigen::Vector3d(tcv.at<double>(0), tcv.at<double>(1), tcv.at<double>(2));

  // collect inlier indices
  out.inlier_idx.reserve(pts1.size());
  for (int i=0;i<(int)pts1.size();++i) if (inliersMask.at<uchar>(i)) out.inlier_idx.push_back(i);

  out.ok = true;
  return out;
}
