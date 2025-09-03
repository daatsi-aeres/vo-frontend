#include "frontend/frontend.h"
#include <iostream>
#include <opencv2/highgui.hpp>   // imshow, waitKey
#include <opencv2/imgproc.hpp>   // cvtColor, line, circle, putText
#ifdef USE_PANGOLIN
#include "visualization/pangolin_visualizer.h"
#endif

Frontend::Frontend(const PinholeCamera& cam)
  : cam_(cam),
    detector_(FeatureDetectorCfg{}),
    matcher_(MatchOptions{}),
    initializer_(InitOptions{}),
    triangulator_(),
    pnp_(PnPOptions{})
#ifdef USE_PANGOLIN
    , vis_()
#endif
{}


// -----------------------------------------------------------------------------
// Initialization (first two frames)
// -----------------------------------------------------------------------------
bool Frontend::initialize(Frame& f1, Frame& f2) 
{
  detector_.detectAndDescribe(f1);
  detector_.detectAndDescribe(f2);

  auto matches = matcher_.match(f1,f2);
  std::vector<cv::Point2f> p1, p2;
  for (auto& m : matches) {
    p1.emplace_back(f1.keypoints[m.i1].pt);
    p2.emplace_back(f2.keypoints[m.i2].pt);
  }

  auto ir = initializer_.estimateRelativePose(p1, p2, cam_);
  if (!ir.ok) return false;

  Eigen::Matrix<double,3,4> P1 = Triangulator::composeP(
      cam_.K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  Eigen::Matrix<double,3,4> P2 = Triangulator::composeP(
      cam_.K, ir.R, ir.t);

  for (int idx : ir.inlier_idx) {
    Eigen::Vector2d uv1(p1[idx].x, p1[idx].y);
    Eigen::Vector2d uv2(p2[idx].x, p2[idx].y);
    auto tr = triangulator_.linear(P1, P2, uv1, uv2);
    if (tr.ok && tr.err1 < 2 && tr.err2 < 2) {
      Landmark lm;
      lm.id = next_landmark_id_++;
      lm.Xw = tr.Xw;
      lm.seen = 2;
      lm.descriptor = f1.descriptors.row(matches[idx].i1).clone();
      landmarks_.push_back(lm);

      f1.kp_to_landmark[matches[idx].i1] = lm.id;
      f2.kp_to_landmark[matches[idx].i2] = lm.id;
    }
  }

  f1.T_cw = Pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  f2.T_cw = Pose(ir.R, ir.t);

  keyframes_.push_back(f1);
  keyframes_.push_back(f2);
  trajectory_.push_back(f1.T_cw);
  trajectory_.push_back(f2.T_cw);

  #ifdef USE_PANGOLIN
  vis_.update(landmarks_, trajectory_, f2.T_cw);
#endif


  std::cout << "Init done: landmarks=" << landmarks_.size() << "\n";
  return true;
}

// -----------------------------------------------------------------------------
// Tracking
// -----------------------------------------------------------------------------
bool Frontend::trackFrame(Frame& f) 
{
  detector_.detectAndDescribe(f);

  Frame& lastkf = keyframes_.back();
  auto matches = matcher_.match(lastkf, f);

  // propagate landmark associations
  for (auto& m : matches) {
    int lm_id = lastkf.kp_to_landmark[m.i1];
    if (lm_id >= 0) {
      f.kp_to_landmark[m.i2] = lm_id;
    }
  }

  // build 2D-3D correspondences
  std::vector<Eigen::Vector3d> Xw;
  std::vector<cv::Point2f> uv;
  for (int i = 0; i < f.kp_to_landmark.size(); ++i) {
    int lm_id = f.kp_to_landmark[i];
    if (lm_id >= 0) {
      Xw.push_back(landmarks_[lm_id].Xw);
      uv.push_back(f.keypoints[i].pt);
    }
  }

  if (Xw.size() < 10) {
    std::cout << "Too few correspondences\n";
    return false;
  }

  auto pr = pnp_.solve(Xw, uv, cam_);
  if (!pr.ok) return false;
  f.T_cw = Pose(pr.R, pr.t);

  // reproject landmarks into frame
  reprojectAndMatch(f, f.T_cw);

  // rebuild correspondences with updated matches
  Xw.clear(); uv.clear();
  for (int i = 0; i < f.kp_to_landmark.size(); ++i) {
    int lm_id = f.kp_to_landmark[i];
    if (lm_id >= 0) {
      Xw.push_back(landmarks_[lm_id].Xw);
      uv.push_back(f.keypoints[i].pt);
    }
  }

  auto pr2 = pnp_.solve(Xw, uv, cam_);
  Pose final_pose = pr2.ok ? Pose(pr2.R, pr2.t) : Pose(pr.R, pr.t);
  f.T_cw = final_pose;

  // update trajectory + Pangolin viz
  trajectory_.push_back(final_pose);
  
  #ifdef USE_PANGOLIN
  vis_.update(landmarks_, trajectory_, final_pose);
  #endif

  std::cout << "Tracked frame with " << pr.inliers.size()
            << " inliers, mean err=" << pr.mean_reproj_err << "\n";

  // --- 2D reprojection debug visualization ---
  cv::Mat vis;
  cv::cvtColor(f.image, vis, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < uv.size(); ++i) {
    const Eigen::Vector3d& X = Xw[i];
    Eigen::Vector3d x_cam = f.T_cw.R * X + f.T_cw.t;
    if (x_cam(2) <= 0) continue;

    cv::Point2f p_proj(
        cam_.K(0,0) * x_cam(0)/x_cam(2) + cam_.K(0,2),
        cam_.K(1,1) * x_cam(1)/x_cam(2) + cam_.K(1,2));

    cv::Point2f p_obs = uv[i];

    cv::circle(vis, p_obs, 2, cv::Scalar(0,0,255), -1);   // red = observed
    cv::circle(vis, p_proj, 2, cv::Scalar(0,255,0), -1);  // green = projected
    cv::line(vis, p_proj, p_obs, cv::Scalar(255,0,0), 1); // blue residual
  }

  cv::putText(vis,
      "Inliers=" + std::to_string(pr.inliers.size()) +
      "  mean err=" + std::to_string(pr.mean_reproj_err) + "px",
      {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);

  cv::imshow("Tracking debug", vis);
  cv::waitKey(1);   // show live video

  if (pr.inliers.size() < 50) {
    insertKeyframe(f);
  }
  return true;
}

// -----------------------------------------------------------------------------
// Keyframe insertion
// -----------------------------------------------------------------------------
void Frontend::insertKeyframe(Frame& f) 
{
  std::cout << "Inserting new keyframe...\n";
  keyframes_.push_back(f);

  Frame& prevkf = keyframes_[keyframes_.size()-2];
  auto matches = matcher_.match(prevkf, f);

  Eigen::Matrix<double,3,4> P1 = Triangulator::composeP(cam_.K,
                             prevkf.T_cw.R, prevkf.T_cw.t);
  Eigen::Matrix<double,3,4> P2 = Triangulator::composeP(cam_.K,
                             f.T_cw.R, f.T_cw.t);

  for (auto& m : matches) {
    Eigen::Vector2d uv1(prevkf.keypoints[m.i1].pt.x, prevkf.keypoints[m.i1].pt.y);
    Eigen::Vector2d uv2(f.keypoints[m.i2].pt.x, f.keypoints[m.i2].pt.y);
    auto tr = triangulator_.linear(P1, P2, uv1, uv2);
    if (tr.ok && tr.err1 < 2 && tr.err2 < 2) {
      Landmark lm;
      lm.id = next_landmark_id_++;
      lm.Xw = tr.Xw;
      lm.seen = 2;
      lm.descriptor = prevkf.descriptors.row(m.i1).clone();
      landmarks_.push_back(lm);

      prevkf.kp_to_landmark[m.i1] = lm.id;
      f.kp_to_landmark[m.i2] = lm.id;
    }
  }
  std::cout << "Map landmarks=" << landmarks_.size() << "\n";
}

// -----------------------------------------------------------------------------
// Descriptor distance (Hamming for ORB)
// -----------------------------------------------------------------------------
int Frontend::descriptorDistance(const cv::Mat& d1, const cv::Mat& d2) {
  const int* pa = d1.ptr<int32_t>();
  const int* pb = d2.ptr<int32_t>();
  int dist = 0;
  for (int i=0; i<8; i++, pa++, pb++) {
    unsigned int v = *pa ^ *pb;
    dist += __builtin_popcount(v);
  }
  return dist;
}


// -----------------------------------------------------------------------------
// Reprojection-based landmark matching
// -----------------------------------------------------------------------------
void Frontend::reprojectAndMatch(Frame& f, const Pose& pose) {
  for (const auto& lm : landmarks_) {
    Eigen::Vector3d x_cam = pose.R * lm.Xw + pose.t;
    if (x_cam(2) <= 0) continue;

    float u = cam_.K(0,0)*x_cam(0)/x_cam(2) + cam_.K(0,2);
    float v = cam_.K(1,1)*x_cam(1)/x_cam(2) + cam_.K(1,2);
    if (u < 0 || u >= f.image.cols || v < 0 || v >= f.image.rows) continue;

    int best_idx = -1;
    int best_dist = 1000;
    for (int i = 0; i < f.keypoints.size(); i++) {
      if (f.kp_to_landmark[i] >= 0) continue;

      cv::Point2f pt = f.keypoints[i].pt;
      if (fabs(pt.x - u) > 10 || fabs(pt.y - v) > 10) continue;

      int dist = descriptorDistance(lm.descriptor, f.descriptors.row(i));
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = i;
      }
    }

    if (best_idx >= 0 && best_dist < 50) {
      f.kp_to_landmark[best_idx] = lm.id;
    }
  }
}
