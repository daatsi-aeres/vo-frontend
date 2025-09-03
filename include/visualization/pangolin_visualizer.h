#pragma once
#include <pangolin/pangolin.h>
#include <vector>
#include <Eigen/Core>
#include "types/frame.h"
#include "types/landmark.h"
#include "core/pose.h"

class PangolinVisualizer {
public:
    PangolinVisualizer();
    void update(const std::vector<Landmark>& landmarks,
                const std::vector<Pose>& trajectory,
                const Pose& current_pose);

    bool shouldStep();   

private:
    pangolin::OpenGlRenderState s_cam_;   // persistent OpenGL render state
    pangolin::View* d_cam_;

    bool step_requested_ = false;
};
