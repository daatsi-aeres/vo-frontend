#include "frontend/frontend.h"
#include <opencv2/opencv.hpp>
#include <iostream>

#ifdef USE_PANGOLIN
#include "visualization/pangolin_visualizer.h"
#include <pangolin/pangolin.h>
#endif

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: ./run_frontend img1 img2 img3 ... fx fy cx cy\n";
        return 1;
    }

    double fx = atof(argv[argc-4]);
    double fy = atof(argv[argc-3]);
    double cx = atof(argv[argc-2]);
    double cy = atof(argv[argc-1]);

    PinholeCamera cam; 
    cam.K << fx, 0, cx,
             0, fy, cy,
             0, 0, 1;

    Frontend fe(cam);

    Frame f1, f2;
    f1.image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    f2.image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if(!fe.initialize(f1, f2)) {
        std::cerr << "Init failed\n"; 
        return 1;
    }

    // --- Process remaining frames step-by-step ---
    for (int i = 3; i < argc-4; ++i) {
        Frame f;
        f.image = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
        if (f.image.empty()) continue;

#ifdef USE_PANGOLIN
        // --- Pangolin mode: wait for SPACE ---
        while (!pangolin::ShouldQuit()) {
            if (fe.vis_.shouldStep()) {
                if(!fe.trackFrame(f)) {
                    std::cerr << "Tracking failed on frame " << i-1 << "\n";
                    return 1;
                }
                break;
            }
            pangolin::FinishFrame();   // keep window responsive
        }
#else
        // --- OpenCV mode: press any key to step ---
        std::cout << "Press any key to process next frame..." << std::endl;
        int key = cv::waitKey(0);   // wait indefinitely
        if (key == 27) break;       // ESC to quit

        if(!fe.trackFrame(f)) {
            std::cerr << "Tracking failed on frame " << i-1 << "\n";
            return 1;
        }
#endif
    }

    return 0;
}
