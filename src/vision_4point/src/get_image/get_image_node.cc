#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.h>

int main(int agrc, char *argv[]) {

  // 1.设置编码格式
  setlocale(LC_ALL, "");
  // 2.初始化 ROS 节点:命名(唯一)
  ros::init(agrc, argv, "get_image_node");
  // 3.实例化 ROS 句柄(该类封装了 ROS 中的一些常用功能)
  ros::NodeHandle nh;
  // 4.实例化发布者对象
  ros::Publisher pub = nh.advertise<sensor_msgs::Image>("camera/image", 10);

  // 使用opencv读取网络相机视频流
  cv::Mat frame;
  cv::VideoCapture cap(0);
  // 获取相机ip参数
  std::string camera_ip;
  nh.getParam("camera_ip", camera_ip);
  std::cout << "camera_ip: " << camera_ip << std::endl;
  // cap.open(camera_ip);
  // 判断是否打开相机
  if (!cap.isOpened()) {
    ROS_INFO("can not open camera");
    return -1;
  }
  // 得到相机分辨率参数，并将其存入ros参数服务器
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  nh.setParam("camera_width", width);
  nh.setParam("camera_height", height);

  int count = 0;
  ros::Rate r(1000);
  while (ros::ok()) {
    //从视频流中读取图片
    cap.read(frame);
    // 从ros参数服务器获取是否标定的参数
    bool is_calibration;
    nh.getParam("is_calibration", is_calibration);
    if (is_calibration) {
      cv::Mat calibration = frame.clone();
      cv::resize(calibration, calibration, cv::Size(720, 540));
      // 实时展示图片，用于标定，当键入c时，保存图片
      cv::imshow("calibration", calibration);
      int key = cv::waitKey(1000);
      if (key == 'c') {
        std::string path =
            "/home/tjurm/Desktop/ros_learn/src/radar_vision/src/data/";
        cv::imwrite(path + "x.jpg", frame);
      }
    }
    //将图片尺寸压缩为720*540
    // cv::resize(frame, frame, cv::Size(720, 540));
    //将图片转换成ROS消息
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub.publish(msg);
    ROS_INFO("image published");
    r.sleep();
    ros::spinOnce();
  }
  return 0;
}