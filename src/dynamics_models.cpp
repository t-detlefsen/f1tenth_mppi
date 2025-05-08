#include "dynamics_models.hpp"

KBM::KBM() {}
KBM::~KBM() {}

std::vector<double> KBM::predict_euler(std::vector<double> state, std::vector<double> action) {
    std::vector<double> x_dot = dynamics(state, action);
    return {state[0] + x_dot[0] * dt_, state[1] + x_dot[1] * dt_, state[2] + x_dot[2] * dt_};
}

std::vector<double> KBM::dynamics(std::vector<double> state, std::vector<double> action) {
    double x_dot = action[0] * std::cos(state[2]);
    double y_dot = action[0] * std::sin(state[2]);
    double theta_dot = action[0] * std::tan(action[1]) / L_;

    return {x_dot, y_dot, theta_dot};
}

void KBM::set_params(double L, double dt) {
    L_ = L;
    dt_ = dt;
}