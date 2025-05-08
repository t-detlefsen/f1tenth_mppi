#include <vector>
#include <cmath>

class KBM {
    private:
        double L_;
        double min_throttle_;
        double max_throttle_;
        double max_steer_;
        double dt_;
    public:
        KBM();
        ~KBM();

        std::vector<double> dynamics(std::vector<double> state, std::vector<double> action);
        std::vector<double> predict_euler(std::vector<double> state, std::vector<double> action);
        void set_params(double L, double dt); 
};