#pragma once
#include <Eigen/Dense>

// Pinhole intrinsics and pixel<->normalized helpers.
struct PinholeCamera {
  // [ fx 0 cx; 0 fy cy; 0 0 1 ]
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  // optional distortion vector (can be empty)
  Eigen::VectorXd dist;

  // pixel (u,v) -> normalized (x,y): x=(u-cx)/fx, y=(v-cy)/fy
  Eigen::Vector2d toNormalized(const Eigen::Vector2d& uv) const;
  // normalized (x,y) -> pixel (u,v)
  Eigen::Vector2d toPixel(const Eigen::Vector2d& xy) const;

  // Projection matrix P = K [R|t]
  Eigen::Matrix<double,3,4> projection(const Eigen::Matrix3d& R,
                                       const Eigen::Vector3d& t) const;
};
