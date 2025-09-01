#include <opencv2/calib3d.hpp>
#include "frontend/pose_estimator.h"

PoseEstimator::PoseEstimator(const PnPOptions& opt) : opt_(opt) {}

PnPResult PoseEstimator::solve(const std::vector<Eigen::Vector3d>& Xw,
                               const std::vector<cv::Point2f>& uv,
                               const PinholeCamera& cam) const {
  PnPResult out;
  if (Xw.size() < 4 || Xw.size() != uv.size()) return out;

  // Pack into cv::Mat for OpenCV
  std::vector<cv::Point3f> obj; obj.reserve(Xw.size());
  for (auto& X : Xw) obj.emplace_back((float)X.x(), (float)X.y(), (float)X.z());

  cv::Mat rvec, tvec, inliersCv;
  cv::Mat Kcv(3,3,CV_64F);
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) Kcv.at<double>(r,c) = cam.K(r,c);
  cv::Mat dist; // assume rectified / no distortion for now

  bool ok = cv::solvePnPRansac(
      obj, uv, Kcv, dist,
      rvec, tvec,
      false, opt_.max_iters,
      opt_.ransac_thresh_px, opt_.conf, inliersCv,
      cv::SOLVEPNP_EPNP // robust and fast; OpenCV may switch internally
  );

  if (!ok || inliersCv.rows < 4) return out;

  // Convert rvec to R
  cv::Mat Rcv;
  cv::Rodrigues(rvec, Rcv);
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) out.R(r,c) = Rcv.at<double>(r,c);
  out.t = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  // Inliers
  out.inliers.reserve(inliersCv.rows);
  for (int i=0;i<inliersCv.rows;++i) out.inliers.push_back(inliersCv.at<int>(i));

  // Mean reprojection error (on inliers)
  double err_sum = 0.0;
  for (int idx : out.inliers) {
    const auto& X = Xw[idx];
    cv::Mat Xh = (cv::Mat_<double>(4,1) << X.x(), X.y(), X.z(), 1.0);
    cv::Mat Rt  = (cv::Mat_<double>(3,4) <<
      out.R(0,0), out.R(0,1), out.R(0,2), out.t(0),
      out.R(1,0), out.R(1,1), out.R(1,2), out.t(1),
      out.R(2,0), out.R(2,1), out.R(2,2), out.t(2));
    cv::Mat xh = Kcv * Rt * Xh;
    double u = xh.at<double>(0)/xh.at<double>(2);
    double v = xh.at<double>(1)/xh.at<double>(2);
    double du = u - uv[idx].x;
    double dv = v - uv[idx].y;
    err_sum += std::sqrt(du*du + dv*dv);
  }
  out.mean_reproj_err = err_sum / std::max(1,(int)out.inliers.size());
  out.ok = true;
  return out;
}
