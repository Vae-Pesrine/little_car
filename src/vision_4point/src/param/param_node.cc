#include "ros/ros.h"
#include <string>

int main(int agrc, char *argv[]){
    setlocale(LC_ALL, "");
    ros::init(agrc, argv, "param_node");
    // 开始设置参数
    ros::NodeHandle nh;
    std::string camera_ip = "/home/tjurm/Desktop/ros_learn/src/radar_vision/assets/test.MP4";
    while(true){
        nh.setParam("camera_ip", camera_ip);
    }
    return 0;
}