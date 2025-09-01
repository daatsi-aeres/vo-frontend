#pragma once
#include <Eigen/Dense>

// // World->Camera pose T_cw = [R|t]
// struct Pose {
//   Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
//   Eigen::Vector3d t = Eigen::Vector3d::Zero();

//   // 4x4 homogeneous matrix [R t; 0 0 0 1]
//   Eigen::Matrix4d matrix() const;
//   // Inverse (Camera->World)
//   Pose inverse() const;
//   // Compose: this ∘ other (apply 'other' then this)
//   Pose compose(const Pose& other) const;
// };

struct Pose {
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  Pose() {
    R = Eigen::Matrix3d::Identity();
    t = Eigen::Vector3d::Zero();
  }

  Pose(const Eigen::Matrix3d& R_, const Eigen::Vector3d& t_) {
    R = R_;
    t = t_;
  }
};
