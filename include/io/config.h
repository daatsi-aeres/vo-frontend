#pragma once
#include <string>
#include <Eigen/Dense>
#include "core/camera.h"

// Front-end parameters (tweak via YAML later)
struct FrontendParams {
  // ORB
  int orb_n_features = 1500;
  // Essential matrix RANSAC
  double ransac_E_thresh_px = 1.0;
  double ransac_conf = 0.999;
  // Triangulation checks
  double max_reproj_err_px = 2.0;
  double min_parallax_deg = 1.0;
  // PnP RANSAC
  double pnp_ransac_thresh_px = 3.0;
  int    pnp_ransac_max_iters = 1000;
};

class Config {
 public:
  virtual ~Config() = default;
  virtual bool load(const std::string& path) = 0;
  virtual const PinholeCamera& camera() const = 0;
  virtual const FrontendParams& params() const = 0;
  virtual std::string dataset_root() const = 0;
};
