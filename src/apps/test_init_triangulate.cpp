#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "frontend/feature_detector.h"
#include "frontend/feature_matcher.h"
#include "frontend/initializer.h"
#include "frontend/triangulator.h"
#include "core/camera.h"

// Uncomment if you have Pangolin installed and linked in CMake
// #define HAS_PANGOLIN 1
#ifdef HAS_PANGOLIN
#include <pangolin/pangolin.h>
#endif

static void savePLY(const std::string& path, const std::vector<Eigen::Vector3d>& pts)
{
  std::ofstream ofs(path);
  if (!ofs.is_open()) { std::cerr << "Failed to open " << path << "\n"; return; }
  ofs << "ply\nformat ascii 1.0\n";
  ofs << "element vertex " << pts.size() << "\n";
  ofs << "property float x\nproperty float y\nproperty float z\nend_header\n";
  for (const auto& p : pts) ofs << (float)p.x() << " " << (float)p.y() << " " << (float)p.z() << "\n";
  ofs.close();
  std::cout << "Saved PLY: " << path << " (" << pts.size() << " points)\n";
}

static Eigen::Vector3d rotToEulerXYZ(const Eigen::Matrix3d& R)
{
  // extrinsic XYZ euler (roll=X, pitch=Y, yaw=Z). Beware of conventions; this is for quick intuition.
  double sy = std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0));
  bool singular = sy < 1e-6;
  double x, y, z;
  if (!singular) {
    x = std::atan2(R(2,1), R(2,2));
    y = std::atan2(-R(2,0), sy);
    z = std::atan2(R(1,0), R(0,0));
  } else {
    x = std::atan2(-R(1,2), R(1,1));
    y = std::atan2(-R(2,0), sy);
    z = 0;
  }
  return {x, y, z};
}

int main(int argc, char** argv){
  if (argc < 7) {
    std::cerr << "Usage: ./test_init_triangulate img1 img2 fx fy cx cy\n";
    return 1;
  }
  cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (img1.empty() || img2.empty()) { std::cerr << "bad images\n"; return 1; }

  double fx = atof(argv[3]), fy = atof(argv[4]);
  double cx = atof(argv[5]), cy = atof(argv[6]);
  PinholeCamera cam;
  cam.K << fx, 0, cx,
           0, fy, cy,
           0,  0,  1;

  // 1) Detect + match
  Frame f1, f2; f1.image = img1; f2.image = img2;
  FeatureDetector detector({});
  detector.detectAndDescribe(f1);
  detector.detectAndDescribe(f2);
  FeatureMatcher matcher({0.8,false,5000});
  auto matches = matcher.match(f1, f2);

  if (matches.size() < 30) {
    std::cerr << "Too few matches: " << matches.size() << "\n";
    return 1;
  }

  std::vector<cv::Point2f> p1, p2;
  p1.reserve(matches.size()); p2.reserve(matches.size());
  std::vector<cv::DMatch> cv_inlier_matches; cv_inlier_matches.reserve(matches.size());
  for (size_t i=0;i<matches.size();++i) {
    p1.emplace_back(f1.keypoints[matches[i].i1].pt);
    p2.emplace_back(f2.keypoints[matches[i].i2].pt);
  }

  // 2) Relative pose via Essential
  Initializer init({1.0, 0.999});
  auto ir = init.estimateRelativePose(p1, p2, cam);
  if (!ir.ok) { std::cerr << "init failed\n"; return 1; }

  // 3) Build P1, P2
  Triangulator tri;
  Eigen::Matrix<double,3,4> P1 = Triangulator::composeP(cam.K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  Eigen::Matrix<double,3,4> P2 = Triangulator::composeP(cam.K, ir.R, ir.t);

  // 4) Triangulate inliers, collect points and a match viz of inliers only
  std::vector<Eigen::Vector3d> cloud;
  cloud.reserve(ir.inlier_idx.size());

  for (int idx : ir.inlier_idx) {
    Eigen::Vector2d uv1(p1[idx].x, p1[idx].y);
    Eigen::Vector2d uv2(p2[idx].x, p2[idx].y);
    auto tr = tri.linear(P1, P2, uv1, uv2);
    if (tr.ok && tr.err1 < 2.0 && tr.err2 < 2.0) {
      cloud.push_back(tr.Xw);
      // keep only inlier match for viz
      cv_inlier_matches.emplace_back(
        /*queryIdx=*/matches[idx].i1,
        /*trainIdx=*/matches[idx].i2,
        /*distance=*/matches[idx].distance
      );
    }
  }

  // 5) Print R and t (t normalized)
  std::cout << "Inliers (E RANSAC): " << ir.inlier_idx.size() << "\n";
  std::cout << "Triangulated OK: " << cloud.size() << "\n\n";

  std::cout << "R =\n" << ir.R << "\n\n";
  Eigen::Vector3d eul = rotToEulerXYZ(ir.R) * 180.0 / M_PI;
  std::cout << "Euler XYZ (deg) ~ roll=" << eul.x()
            << " pitch=" << eul.y()
            << " yaw=" << eul.z() << "\n";

  Eigen::Vector3d t_unit = ir.t.normalized();
  std::cout << "t (unit, scale-ambiguous) = [" << t_unit.transpose() << "]^T\n\n";

  // 6) Show inlier matches visualization
  cv::Mat vis;
  cv::drawMatches(img1, f1.keypoints, img2, f2.keypoints, cv_inlier_matches, vis);
  cv::imshow("Inlier matches (E)", vis);
  cv::waitKey(1); // keep window responsive if Pangolin is used below

  // 7) Save PLY always
  savePLY("landmarks_init.ply", cloud);

#ifdef HAS_PANGOLIN
  // 8) Optional: live 3D viewer
  pangolin::CreateWindowAndBind("Init Triangulation - 3D", 1024, 768);
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState cam_state(
      pangolin::ProjectionMatrix(1024,768, 500,500, 512,384, 0.1, 1000),
      pangolin::ModelViewLookAt(0,-5,2, 0,0,0, 0,0,1)
  );
  pangolin::View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0)
      .SetHandler(new pangolin::Handler3D(cam_state));

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(cam_state);

    // draw world axes at origin
    glLineWidth(2.f);
    glBegin(GL_LINES);
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0); // X
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0); // Y
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1); // Z
    glEnd();

    // draw camera 1 at origin (tiny frustum)
    glColor3f(1,1,0);
    glBegin(GL_LINES);
    float s = 0.2f;
    glVertex3f(0,0,0); glVertex3f( s, s, 1*s);
    glVertex3f(0,0,0); glVertex3f(-s, s, 1*s);
    glVertex3f(0,0,0); glVertex3f( s,-s, 1*s);
    glVertex3f(0,0,0); glVertex3f(-s,-s,1*s);
    glEnd();

    // draw points
    glPointSize(2.f);
    glColor3f(1,1,1);
    glBegin(GL_POINTS);
    for (const auto& p : cloud) glVertex3f((float)p.x(), (float)p.y(), (float)p.z());
    glEnd();

    pangolin::FinishFrame();
    // also keep OpenCV window alive
    int key = cv::waitKey(1);
    (void)key;
  }
#else
  std::cout << "Tip: For a live 3D view, build with Pangolin and define HAS_PANGOLIN.\n";
  std::cout << "     Otherwise open landmarks_init.ply in MeshLab or CloudCompare.\n";
  std::cout << "Press any key in the image window to exit...\n";
  cv::waitKey(0);
#endif

  return 0;
}
