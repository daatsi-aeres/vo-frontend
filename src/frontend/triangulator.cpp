#include <Eigen/Dense>
#include "frontend/triangulator.h"

static inline Eigen::Vector2d reproj(const Eigen::Matrix<double,3,4>& P,
                                     const Eigen::Vector3d& X) {
  Eigen::Vector4d Xh; Xh << X, 1.0;
  Eigen::Vector3d x = P * Xh;
  return {x(0)/x(2), x(1)/x(2)};
}

TriResult Triangulator::linear(const Eigen::Matrix<double,3,4>& P1,
                               const Eigen::Matrix<double,3,4>& P2,
                               const Eigen::Vector2d& uv1,
                               const Eigen::Vector2d& uv2) const {
  TriResult r;
  Eigen::RowVector4d p11 = P1.row(0), p12 = P1.row(1), p13 = P1.row(2);
  Eigen::RowVector4d p21 = P2.row(0), p22 = P2.row(1), p23 = P2.row(2);

  Eigen::Matrix4d A;
  A.row(0) = uv1.x() * p13 - p11;
  A.row(1) = uv1.y() * p13 - p12;
  A.row(2) = uv2.x() * p23 - p21;
  A.row(3) = uv2.y() * p23 - p22;

  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4d Xh = svd.matrixV().col(3);
  if (std::abs(Xh[3]) < 1e-12) return r;

  Eigen::Vector3d X = Xh.head<3>() / Xh[3];

  // reprojection error in pixels
  Eigen::Vector2d u1h = reproj(P1, X);
  Eigen::Vector2d u2h = reproj(P2, X);
  r.err1 = (u1h - uv1).norm();
  r.err2 = (u2h - uv2).norm();

  // cheirality: positive depth in both cams
  auto depth = [&](const Eigen::Matrix<double,3,4>& P){
    Eigen::Vector4d XH; XH << X, 1.0;
    return (P * XH)(2);
  };
  if (depth(P1) > 0 && depth(P2) > 0) {
    r.ok = true;
    r.Xw = X;
  }
  return r;
}
