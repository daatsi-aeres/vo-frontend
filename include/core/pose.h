// pose.h
struct Pose {
Eigen::Matrix3d R; // world->camera rotation
Eigen::Vector3d t; // world->camera translation
Eigen::Matrix4d matrix() const; // [R|t;0 0 0 1]
Pose inverse() const; // camera->world
};