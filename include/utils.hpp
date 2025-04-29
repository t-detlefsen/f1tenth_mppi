#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "ament_index_cpp/get_package_prefix.hpp"

std::vector<std::vector<double>> load_waypoints(const std::string &file_path);

void publish_markers(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher, std::vector<std::vector<double>> points);

void publish_trajectory(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher, std::vector<std::vector<double>> points);

double clamp(double num, double min, double max);

double get_dist(std::vector<double> p1, std::vector<double> p2);