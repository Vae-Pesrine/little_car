#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// 串口相关
#include <sensor_msgs/image_encodings.h>
#include <serial/serial.h>
#include <string>

// zbar相关

int main(int agrc, char *argv[]) {

  // 1.设置编码格式
  setlocale(LC_ALL, "");
  // 2.初始化 ROS 节点:命名(唯一)
  ros::init(agrc, argv, "control_node");
  // 3.实例化 ROS 句柄(该类封装了 ROS 中的一些常用功能)
  ros::NodeHandle nh;

  // 初始化串口
  serial::Serial ser;
  ser.setPort("/dev/ttyUSB0");
  ser.setBaudrate(9600);
  // ser.available();
  serial::Timeout to = serial::Timeout::simpleTimeout(10);
  ser.setTimeout(to);
  ser.open();
  ser.write("1");
  // ser.write(123,3);
  //   std::string a;
  // a = ser.readline(10000, "b");
  //   a = ser.readline(1000, " ");
  //   std::cout << a << std::endl;

  return 0;
}