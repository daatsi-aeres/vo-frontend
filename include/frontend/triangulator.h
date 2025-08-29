#pragma once
#include <Eigen/Dense>

struct TriResult {
  bool ok = false;
  Eigen::Vector3d Xw;
  double err1 = -1, err2 = -1;
};

class Triangulator {
 public:
  // Linear SVD triangulation from two views with P=K[R|t] (pixel domain)
  TriResult linear(const Eigen::Matrix<double,3,4>& P1,
                   const Eigen::Matrix<double,3,4>& P2,
                   const Eigen::Vector2d& uv1,
                   const Eigen::Vector2d& uv2) const;

  static inline Eigen::Matrix<double,3,4> composeP(const Eigen::Matrix3d& K,
                                                   const Eigen::Matrix3d& R,
                                                   const Eigen::Vector3d& t) {
    Eigen::Matrix<double,3,4> Rt; Rt.block<3,3>(0,0)=R; Rt.col(3)=t; return K*Rt;
  }
};
