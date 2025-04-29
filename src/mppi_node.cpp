#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream> 
#include <stdexcept>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "utils.hpp"
#include "dynamics_models.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/buffer.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "ament_index_cpp/get_package_prefix.hpp"
#include <visualization_msgs/msg/marker.hpp>

/// CHECK: include needed ROS msg type headers and libraries

using std::placeholders::_1;
using std::placeholders::_2;
#define rad2deg 180.0 / M_PI
#define deg2rad M_PI / 180

class MPPI : public rclcpp::Node
{

private:
    // Publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // Subscribers
    message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> pose_sub_;
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;

    // Transform objects
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Other
    std::vector<std::vector<double>> waypoints;
    KBM model;
    nav_msgs::msg::OccupancyGrid og = nav_msgs::msg::OccupancyGrid();
    std::vector<std::vector<double>> u_prev;

public:
    MPPI() : Node("mppi_node") {
        /// TODO: Setup Loggers

        // Initialize Parameters
        initialize_parameters();

        // Initialize Model
        model.set_params(this->get_parameter("wheelbase").as_double(),
                         this->get_parameter("dt").as_double());

        // Initialize Occupancy Grid
        og.header.frame_id = this->get_parameter("vehicle_frame").as_string();
        og.info.resolution = this->get_parameter("cost_map_res").as_double();
        og.info.width = this->get_parameter("cost_map_width").as_int();
        og.info.height = this->get_parameter("cost_map_width").as_int();
        og.info.origin.position.x = 0.0;
        og.info.origin.position.y = -og.info.height * og.info.resolution / 2;

        // Load Waypoints
        try {
            waypoints = load_waypoints(this->get_parameter("waypoint_path").as_string());
            /// TODO: Normalize speed to min + max
        } catch (const std::runtime_error& error) {
            std::cout << "Issue loading waypoints\n";
            std::cerr << error.what() << std::endl;
        }
        
        // Initialize Previous Control Input Sequence
        for (int i = 0; i < this->get_parameter("steps_trajectories").as_int(); i++) {
            std::vector<double> zeros = {0.0, 0.0};
            u_prev.push_back(zeros);
        }

        // Setup TF2 Buffer + Listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Create Publishers
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(this->get_parameter("drive_topic").as_string(), 10);
        occupancy_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>(this->get_parameter("occupancy_topic").as_string(), 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->get_parameter("marker_topic").as_string(), 10);

        // Visualize Waypoints
        publish_markers(marker_pub_, waypoints);

        // Create message_filter Subscribers
        scan_sub_.subscribe(this, this->get_parameter("scan_topic").as_string());
        pose_sub_.subscribe(this, this->get_parameter("pose_topic").as_string());

        // Create time synchronized callback
        sync_ = std::make_shared<Sync>(SyncPolicy(10), scan_sub_, pose_sub_);
        sync_->registerCallback(std::bind(&MPPI::callback, this, _1, _2));
    }

    void initialize_parameters() {
        this->declare_parameter("visualize", false);
        this->declare_parameter("waypoint_path", "aims_lockers.csv");
        this->declare_parameter("vehicle_frame", "ego_racecar/laser");
        this->declare_parameter("drive_topic", "/drive");
        this->declare_parameter("occupancy_topic", "/occupancy");
        this->declare_parameter("marker_topic", "/trajectory");
        this->declare_parameter("pose_topic", "/ego_racecar/odom");
        this->declare_parameter("scan_topic", "/scan");
        this->declare_parameter("wheelbase", 0.33);
        this->declare_parameter("min_throttle", 0.5);
        this->declare_parameter("max_throttle", 1.5);
        this->declare_parameter("max_steer", 0.4139);
        this->declare_parameter("dt", 0.1);
        this->declare_parameter("num_trajectories", 50);
        this->declare_parameter("steps_trajectories", 25);
        this->declare_parameter("v_sigma", 0.05);
        this->declare_parameter("omega_sigma", 0.2);
        this->declare_parameter("lambda", 10.0);
        this->declare_parameter("cost_map_width", 100);
        this->declare_parameter("cost_map_res", 0.05);
        this->declare_parameter("occupancy_dilation", 10);
    }

    void callback(const std::shared_ptr<const sensor_msgs::msg::LaserScan> scan_msg,
                  const std::shared_ptr<const nav_msgs::msg::Odometry> pose_msg)
    {
        /// TODO: Create Occupancy Grid
        std::cout << "HERE" << std::endl;

        // Get random seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);

        // ---------- MPPI ----------
        // Set Initial State
        tf2::Quaternion q(
            pose_msg->pose.pose.orientation.x,
            pose_msg->pose.pose.orientation.y,
            pose_msg->pose.pose.orientation.z,
            pose_msg->pose.pose.orientation.w
        );
        
        // Convert quaternion to roll, pitch, yaw
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        if (yaw < 0) {yaw += 2 * M_PI;}

        std::vector<double> x0 = {pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  yaw};

        // Load Previous Control Input Sequence
        std::vector<std::vector<double>> u = u_prev;
        
        // Initialize Noise
        std::normal_distribution<double> v_noise(0.0, this->get_parameter("v_sigma").as_double());
        std::normal_distribution<double> omega_noise(0.0, this->get_parameter("omega_sigma").as_double());
        std::vector<std::vector<std::vector<double>>> epsilon; // num_trajectories x steps_trajectories x 2

        // Initialize State Cost
        std::vector<double> S;

        // Loop through timesteps
        for (int i = 0; i < this->get_parameter("num_trajectories").as_int(); i++) {
            // Reset
            std::vector<double> x = x0;
            std::vector<std::vector<double>> epsilon_i;
            double v;
            S.push_back(0.0);
            for (int j = 1; j <= this->get_parameter("steps_trajectories").as_int(); j++) {
                // Add noise
                v = clamp(u[j - 1][0] + v_noise(generator), this->get_parameter("min_throttle").as_double(), this->get_parameter("max_throttle").as_double());
                double omega = clamp(u[j - 1][1] + omega_noise(generator), -this->get_parameter("max_steer").as_double(), this->get_parameter("max_steer").as_double());
                epsilon_i.push_back({v - u[j - 1][0], omega - u[j - 1][1]});

                // Update State
                x = model.predict_euler(x, {v, omega});

                // Add Stage Cost
                S[i] += compute_cost(x, v);
            }

            // Add terminal Cost
            /// TODO: Add magic number to parameters or modify
            S[i] += compute_cost(x, v) * 10.0;
            
            epsilon.push_back(epsilon_i);
        }

        // Compute information theoretic weights
        std::vector<double> w = compute_weights(S);

        // Calculate Added Noise
        std::vector<std::vector<double>> w_epsilon;
        for (int i = 0; i < this->get_parameter("steps_trajectories").as_int(); i++) {
            w_epsilon.push_back({0.0, 0.0});
            for (int j = 0; j < this->get_parameter("num_trajectories").as_int(); j++) {
                w_epsilon[i][0] += w[j] * epsilon[j][i][0];
                w_epsilon[i][1] += w[j] * epsilon[j][i][1];
            }
        }

        /// TODO: Smooth added noise

        // Update + Clip Control Input Sequence
        for (int i = 0; i < this->get_parameter("steps_trajectories").as_int(); i++) {
            u[i][0] += w_epsilon[i][0];
            u[i][1] += w_epsilon[i][1];

            u[i][0] = clamp(u[i][0], this->get_parameter("min_throttle").as_double(), this->get_parameter("max_throttle").as_double());
            u[i][1] = clamp(u[i][1], -this->get_parameter("max_steer").as_double(), this->get_parameter("max_steer").as_double());
        }

        u_prev = u;
        // --------------------------

        // Publish Trajectory
        std::vector<std::vector<double>> x_traj;
        x_traj.push_back(x0);
        for (int i = 1; i < this->get_parameter("steps_trajectories").as_int(); i++) {
            x_traj.push_back(model.predict_euler(x_traj[i-1], u[i]));
        }

        /// TODO: Figure out why visualization is weird
        publish_trajectory(marker_pub_, x_traj);

        // Publish AckermannDriveStamped Message
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.header.stamp = this->get_clock()->now();
        drive_msg.drive.speed = u[0][0];
        drive_msg.drive.steering_angle = u[0][1];
        // drive_pub_->publish(drive_msg);
    }

    double compute_cost(std::vector<double> x_t, double v_t) {
        // Set Stage Cost weights
        /// TODO: Add to parameters
        std::vector<double> stage_cost_weights = {13.5, 13.5, 5.5, 5.5};
        
        // Find nearest waypoint
        std::vector<double> x_ref = get_nearest_waypoint(x_t[0], x_t[1]);

        // Fix yaw
        x_ref[3] += M_PI / 2; // Fix waypoint offset
        if (x_ref[2] < 0) {x_ref[2] += 2 * M_PI;} // -2 * M_PI < x_ref[2] < 2 * M_PI
        if (x_ref[2] - x_t[2] > 4.5) {x_ref[2] -= 2 * M_PI;} // Bring x_t[2] & x_ref[2] together
        else if (x_ref[2] - x_t[2] < -4.5) {x_ref[2] += 2 * M_PI;} // Bring x_t[2] & x_ref[2] together

        // Compute stage cost
        double stage_cost = stage_cost_weights[0] * std::pow(x_t[0] - x_ref[0], 2)
                          + stage_cost_weights[1] * std::pow(x_t[1] - x_ref[1], 2)
                          + stage_cost_weights[2] * std::pow(x_t[2] - x_ref[2], 2)
                          + stage_cost_weights[3] * std::pow(v_t - x_ref[3], 2);

        return stage_cost;
    }

    std::vector<double> get_nearest_waypoint(double x, double y) {
        float closest_dist = INFINITY;
        int closest_idx = 0;

        std::vector<double> x_t = {x, y};
        for (unsigned long i = 0; i < waypoints.size(); i++) {
            std::vector<double> point = {waypoints[i][0], waypoints[i][0]};
            float dist = get_dist(x_t, point);

            if (dist < closest_dist) {
                closest_idx = i;
                closest_dist = dist;
            }
        }

        return waypoints[closest_idx];
    }
    
    std::vector<double> compute_weights(std::vector<double> S) {
        double rho = *std::min_element(S.begin(), S.end());

        double eta = 0;
        for (int i = 0; i < this->get_parameter("num_trajectories").as_int(); i++) {
            eta += exp(-1.0 / this->get_parameter("lambda").as_double()) * (S[i] - rho);
        }

        std::vector<double> w;
        for (int i = 0; i < this->get_parameter("num_trajectories").as_int(); i++) {
            w.push_back((1.0 / eta) * exp(-1.0 / this->get_parameter("lambda").as_double()) * (S[i] - rho));
        }

        return w;
    }

    ~MPPI() {}
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPI>());
    rclcpp::shutdown();
    return 0;
}