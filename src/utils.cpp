#include "utils.hpp"

std::vector<std::vector<double>> load_waypoints(const std::string &file_path) {
    // Get file location
    std::string package_name = "f1tenth_mppi";
    std::string package_path = ament_index_cpp::get_package_prefix(package_name);
    size_t found = package_path.find("install");

    // Load file
    std::string text, token;
    std::ifstream input{package_path.substr(0, found) + "src/f1tenth_mppi/config/" + file_path};

    // Skip first three lines
    for (long i = 0; i < 3; i++) {std::getline (input, text);}

    // Load waypoints
    std::vector<std::vector<double>> waypoints;
    while (std::getline (input, text)) {
        std::stringstream ss(text);
        std::vector<double> point;
        int i = 0;
        while (std::getline(ss, token, ';')) {
            if (i == 1 || i == 2 || i == 3 || i == 5) {
            point.push_back(std::stof(token));
            }
            i += 1;
        }
        waypoints.push_back(point);
    }

    return waypoints;
}

void publish_markers(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_, std::vector<std::vector<double>> points) {
    // Generate MarkerArray message
    visualization_msgs::msg::MarkerArray marker_array;
    for (unsigned long i = 0; i < points.size(); i++) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.ns = "waypoints";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = points[i][0];
        marker.pose.position.y = points[i][1];
        marker.pose.position.z = 0.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.r = 1.0f;
        marker.color.a = 1.0f;

        marker_array.markers.push_back(marker);
    }

    publisher_->publish(marker_array);
}

void publish_trajectory(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_, std::vector<std::vector<double>> points) {
    visualization_msgs::msg::MarkerArray marker_array;
    for (unsigned long i = 1; i < points.size(); i++) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.ns = "opt";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.01;
        marker.color.r = 1.0f;
        marker.color.a = 1.0f;

        geometry_msgs::msg::Point p1, p2;
        p1.x = points[i - 1][0];
        p1.y = points[i - 1][1];
        p2.x = points[i][0];
        p2.y = points[i][1];

        marker.points.push_back(p1);
        marker.points.push_back(p2);

        marker_array.markers.push_back(marker);
    }

    publisher_->publish(marker_array);
}

double clamp(double num, double min, double max) {
    double out = num;
    out = (out > min) ? out : min;
    out = (out < max) ? out : max;

    return out;
}

double get_dist(std::vector<double> p1, std::vector<double> p2) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    return pow(pow(dx, 2) + pow(dy, 2), 0.5);
}