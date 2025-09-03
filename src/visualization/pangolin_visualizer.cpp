#include "visualization/pangolin_visualizer.h"
#include <pangolin/pangolin.h>
#include <Eigen/Core>

// -----------------------------------------------------------------------------
// Constructor: setup Pangolin window + camera
// -----------------------------------------------------------------------------
PangolinVisualizer::PangolinVisualizer()
    : s_cam_(
          pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
          pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin::AxisY)
      ),
      step_requested_(false)
{
    pangolin::CreateWindowAndBind("Main", 1024, 768);
    glEnable(GL_DEPTH_TEST);

    d_cam_ = &pangolin::Display("3d")
                 .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                 .SetHandler(new pangolin::Handler3D(s_cam_));

    // Register spacebar (ASCII 32) for step mode
    pangolin::RegisterKeyPressCallback(
        32,
        [&]() { step_requested_ = true; }
    );
}

// -----------------------------------------------------------------------------
// Update visualization with landmarks, trajectory, and current pose
// -----------------------------------------------------------------------------
void PangolinVisualizer::update(const std::vector<Landmark>& landmarks,
                                const std::vector<Pose>& trajectory,
                                const Pose& current_pose) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    s_cam_.Apply();

    pangolin::glDrawAxis(1.0);

    // Landmarks
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (const auto& lm : landmarks) {
        glVertex3d(lm.Xw(0), lm.Xw(1), lm.Xw(2));
    }
    glEnd();

    // Trajectory
    glLineWidth(2.0f);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINE_STRIP);
    for (const auto& pose : trajectory) {
        Eigen::Vector3d t = pose.t;
        glVertex3d(t(0), t(1), t(2));
    }
    glEnd();
}


// -----------------------------------------------------------------------------
// Step control: return true only if spacebar was pressed
// -----------------------------------------------------------------------------
bool PangolinVisualizer::shouldStep() {
    if (step_requested_) {
        step_requested_ = false;
        return true;
    }
    return false;
}
