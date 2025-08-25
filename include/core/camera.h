// camera.h
struct PinholeCamera {
Eigen::Matrix3d K; // intrinsics
Eigen::VectorXd dist; // distortion (optional)
Eigen::Vector2d normalize(const Eigen::Vector2d& uv) const; // px -> normalized
Eigen::Vector2d pixel(const Eigen::Vector2d& xy) const; // normalized -> px
};