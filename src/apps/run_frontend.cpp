#include <iostream>
#include <opencv2/opencv.hpp>
#include "frontend/frontend.h"

int main(int argc, char** argv) {
  if (argc < 8) {
    std::cerr << "Usage: ./run_frontend img1 img2 img3 ... fx fy cx cy\n";
    return 1;
  }

  double fx = atof(argv[argc-4]);
  double fy = atof(argv[argc-3]);
  double cx = atof(argv[argc-2]);
  double cy = atof(argv[argc-1]);

  PinholeCamera cam; cam.K << fx,0,cx, 0,fy,cy, 0,0,1;
  Frontend fe(cam);

  Frame f1,f2;
  f1.image = cv::imread(argv[1],0);
  f2.image = cv::imread(argv[2],0);
  if(!fe.initialize(f1,f2)) {
    std::cerr << "Init failed\n"; return 1;
  }

  for (int i=3; i<argc-4; ++i) {
    Frame f;
    f.image = cv::imread(argv[i],0);
    if(f.image.empty()) continue;
    if(!fe.trackFrame(f)) {
      std::cerr << "Tracking failed on frame " << i-1 << "\n";
      break;
    }
  }
  return 0;
}
