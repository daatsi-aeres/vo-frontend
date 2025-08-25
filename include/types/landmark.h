// landmark.h
struct Landmark {
int id;
Eigen::Vector3d Xw; // 3D point in world
int seen = 0;
int inliers = 0;
bool bad = false;
};