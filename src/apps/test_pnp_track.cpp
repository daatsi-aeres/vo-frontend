#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "frontend/feature_detector.h"
#include "frontend/feature_matcher.h"
#include "frontend/initializer.h"
#include "frontend/triangulator.h"
#include "frontend/pose_estimator.h"
#include "core/camera.h"
#include "types/frame.h"

struct Assoc {
  // map f2 keypoint index -> landmark index in 'landmarks'
  // we’ll fill it for inlier matches used to triangulate
  std::vector<int> kp2_to_lm;
};

int main(int argc, char** argv){
  if (argc < 8) {
    std::cerr << "Usage: ./test_pnp_track img1 img2 img3 fx fy cx cy\n";
    return 1;
  }
  cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  cv::Mat img3 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
  if (img1.empty()||img2.empty()||img3.empty()){ std::cerr<<"bad images\n"; return 1; }

  double fx = atof(argv[4]), fy = atof(argv[5]);
  double cx = atof(argv[6]), cy = atof(argv[7]);
  PinholeCamera cam;
  cam.K << fx,0,cx, 0,fy,cy, 0,0,1;

  // Detect & match (1↔2)
  Frame f1, f2, f3; f1.image=img1; f2.image=img2; f3.image=img3;
  FeatureDetector detector({});
  detector.detectAndDescribe(f1);
  detector.detectAndDescribe(f2);
  FeatureMatcher matcher({});
  auto m12 = matcher.match(f1, f2);

  // Build pixel vectors for init
  std::vector<cv::Point2f> p1, p2;
  p1.reserve(m12.size()); p2.reserve(m12.size());
  for (auto& m : m12) {
    p1.emplace_back(f1.keypoints[m.i1].pt);
    p2.emplace_back(f2.keypoints[m.i2].pt);
  }

  // Two-view init (E + recoverPose)
  Initializer init({});
  auto ir = init.estimateRelativePose(p1, p2, cam);
  if (!ir.ok) { std::cerr<<"init failed\n"; return 1; }

  // Triangulate inliers -> landmarks
  Triangulator tri;
  Eigen::Matrix<double,3,4> P1 = Triangulator::composeP(cam.K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  Eigen::Matrix<double,3,4> P2 = Triangulator::composeP(cam.K, ir.R, ir.t);

  std::vector<Eigen::Vector3d> landmarks; landmarks.reserve(ir.inlier_idx.size());
  Assoc assoc; assoc.kp2_to_lm.assign((size_t)f2.keypoints.size(), -1);

  for (int k=0; k<(int)ir.inlier_idx.size(); ++k) {
    int idx = ir.inlier_idx[k];
    Eigen::Vector2d uv1(p1[idx].x, p1[idx].y);
    Eigen::Vector2d uv2(p2[idx].x, p2[idx].y);
    auto tr = tri.linear(P1, P2, uv1, uv2);
    if (tr.ok && tr.err1 < 2.0 && tr.err2 < 2.0) {
      int lm_id = (int)landmarks.size();
      landmarks.push_back(tr.Xw);
      // remember which f2 keypoint corresponds to this landmark
      int kp2 = m12[idx].i2;
      assoc.kp2_to_lm[kp2] = lm_id;
    }
  }
  std::cout << "Init landmarks: " << landmarks.size() << "\n";
  if (landmarks.size() < 20) { std::cerr<<"too few landmarks\n"; return 1; }

  // Detect on frame 3 and match (2↔3)
  detector.detectAndDescribe(f3);
  auto m23 = matcher.match(f2, f3);

  // Build 2D–3D correspondences using associations from frame 2
  std::vector<Eigen::Vector3d> Xw;
  std::vector<cv::Point2f>     u3;
  Xw.reserve(m23.size()); u3.reserve(m23.size());
  for (auto& m : m23) {
    int kp2 = m.i1;           // index in f2
    int kp3 = m.i2;           // index in f3
    int lm_id = assoc.kp2_to_lm[kp2];
    if (lm_id >= 0) { // only if this kp2 came from a triangulated landmark
      Xw.push_back(landmarks[lm_id]);
      u3.emplace_back(f3.keypoints[kp3].pt);
    }
  }
  std::cout << "2D-3D for PnP: " << Xw.size() << "\n";
  if (Xw.size() < 20) { std::cerr<<"too few 2D-3D\n"; return 1; }

  // PnP solve for frame 3 pose
  PoseEstimator pnp({});
  auto pr = pnp.solve(Xw, u3, cam);
  if (!pr.ok) { std::cerr<<"PnP failed\n"; return 1; }

  std::cout << "PnP inliers: " << pr.inliers.size()
            << "  mean reproj err: " << pr.mean_reproj_err << " px\n";
  std::cout << "R3=\n" << pr.R << "\n";
  std::cout << "t3^T= [" << pr.t.transpose() << "]\n";

  // (Better) visualize projected landmarks vs. observed keypoints on frame 3
// - green  = projected landmark position using PnP pose (prediction)
// - red    = observed matched keypoint in frame 3 (measurement)
// - blue line connects prediction -> measurement (residual)
 cv::Mat vis;
  // Build K for projection
  cv::Mat Kcv = (cv::Mat_<double>(3,3) << fx, 0, cx,
                                          0, fy, cy,
                                          0,  0,  1);
  // Convert R,t to OpenCV types for projectPoints
  cv::Mat Rcv = (cv::Mat_<double>(3,3) <<
      pr.R(0,0), pr.R(0,1), pr.R(0,2),
      pr.R(1,0), pr.R(1,1), pr.R(1,2),
      pr.R(2,0), pr.R(2,1), pr.R(2,2));
  cv::Mat rvec; cv::Rodrigues(Rcv, rvec);
  cv::Mat tvec = (cv::Mat_<double>(3,1) << pr.t(0), pr.t(1), pr.t(2));

  // Prepare a color image of frame 3
  cv::cvtColor(img3, vis, cv::COLOR_GRAY2BGR);

  // We have correspondences Xw[i] <-> u3[i]. Project each Xw[i] with (R3,t3)
  double err_sum = 0.0;
  for (size_t i = 0; i < Xw.size(); ++i) {
    std::vector<cv::Point3f> obj = {
      cv::Point3f((float)Xw[i].x(), (float)Xw[i].y(), (float)Xw[i].z())
    };
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, Kcv, cv::Mat(), proj);

    // Draw: projected (green), observed (red), residual (blue)
    const cv::Point2f& p_proj = proj[0];
    const cv::Point2f& p_obs  = u3[i];

    // residual line
    cv::line(vis, p_proj, p_obs, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    // projected point
    cv::circle(vis, p_proj, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    // observed keypoint
    cv::circle(vis, p_obs, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

    // accumulate error for display
    double du = p_proj.x - p_obs.x, dv = p_proj.y - p_obs.y;
    err_sum += std::sqrt(du*du + dv*dv);
  }

  // Put a small legend and mean residual text
  const double mean_err = err_sum / std::max<size_t>(1, Xw.size());
  cv::putText(vis, "green=projected landmark, red=observed kp, blue=residual",
              {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,255,50), 1, cv::LINE_AA);
  cv::putText(vis, ("pairs=" + std::to_string(Xw.size()) +
                    "  mean residual(px)=" + std::to_string(mean_err)),
              {10, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,255,50), 1, cv::LINE_AA);

  cv::imshow("Frame 3: projected landmarks vs observed kps", vis);
  cv::waitKey(0);


  cv::imshow("Projected landmarks on frame 3", vis);
  cv::waitKey(0);

  return 0;
}
