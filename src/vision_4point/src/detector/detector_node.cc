// ros
#include "ros/ros.h"

// opencv
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

// msg
#include "vision_4point/circle_position.h"
#include "vision_4point/circle_positions.h"
#include <sensor_msgs/image_encodings.h>

// work
#include "detector/infer/infer.h"
#include "detector/infer/trt.h"
#include <vector>

ros::Publisher pub;
std::shared_ptr<detector::Detector> detector_;
void detect(cv::Mat &src);

void call_detect(const sensor_msgs::Image::ConstPtr &msg) {
  // 展示收到的图片!
  cv::Mat frame;
  frame = cv_bridge::toCvShare(msg, "bgr8")->image;
  bool is_show;
  ROS_INFO("image received");
  // 开始对图片进行处理，得到检测结果并发布
  detect(frame);
}

int main(int agrc, char *argv[]) {
  setlocale(LC_ALL, "");
  ros::init(agrc, argv, "detector_node");
  ros::NodeHandle nh;
  detector_ = std::make_shared<detector::TRT>();
  detector_->init();
  // 订阅图像话题
  ros::Subscriber sub =
      nh.subscribe<sensor_msgs::Image>("camera/image", 1, call_detect);
  // 设置发布检测结果的话题
  pub = nh.advertise<vision_4point::circle_positions>("/circle_positions", 10);
  ros::spin();
  return 0;
}

/*
 * @brief: 识别函数
 * @param: src: 输入图像
 * @return: armors: 识别到的装甲板以及类型（circle_positions消息类型）
 * ros中的消息类型不能作为函数的返回值，所以我们直接在函数内部发布消息！
 */
void detect(cv::Mat &src) {

  bool is_network;

  /* 神经网络初筛部分
   *  注意，目标检测网络的输出可以有多个类
   *  故为了增强推理代码的可扩展性，detect函数的返回默认有多个类
   */
  std::vector<std::vector<detector::Object>> armors_cars =
      detector_->detect(src);
  std::vector<detector::Object> objs = armors_cars[0];
  // detector::utils::ShowObjects(src, objs);
  detector::utils::ShowPoints(src, objs);
  bool is_show = ros::param::get("is_show", is_show);
  if (is_show) {
    cv::imshow("yolov5", src);
    cv::waitKey(1);
  }

  //结合真实尺寸得到的世界坐标系下的坐标，进行solvepnp解算，得到相机坐标系和世界坐标系之间的转换关系
  // 1.得到世界坐标系下的坐标
  /*世界坐标系下特征点坐标*/
  float circle_r = 51.0;
  std::vector<cv::Point3f> object_points;
  object_points.emplace_back(cv::Point3f(circle_r, 0, 0));
  object_points.emplace_back(cv::Point3f(0, -circle_r, 0));
  object_points.emplace_back(cv::Point3f(-circle_r, 0, 0));
  object_points.emplace_back(cv::Point3f(0, circle_r, 0));
  vision_4point::circle_positions points;
  // 2.得到相机坐标系下的坐标
  for (auto obj : objs) {
    std::vector<cv::Point2f> image_points;
    image_points.emplace_back(obj.point[0]);
    image_points.emplace_back(obj.point[1]);
    image_points.emplace_back(obj.point[2]);
    image_points.emplace_back(obj.point[3]);
    // 3.得到相机坐标系和世界坐标系之间的转换关系
    cv::Mat rvec, tvec;
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 446.28399898, 0.0, 365.29741494, 0.0,
         445.96114536, 215.54356105, 0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 0.03597615, -0.05577474,
                           -0.00071241, -0.00039876, 0.03168604);
    cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec,
                 tvec);
    vision_4point::circle_position point;
    // point.x = obj.box.x + obj.box.width / 2;
    // point.y = obj.box.y + obj.box.height / 2;
    point.x = tvec.ptr<double>(0)[0];
    point.y = tvec.ptr<double>(0)[1];
    point.z = tvec.ptr<double>(0)[2];
    point.id = obj.id;
    points.data.push_back(point);
    // std::cout << "position_x: " << tvec.ptr<double>(0)[0] << std::endl;
    // std::cout << "position_y: " << tvec.ptr<double>(0)[1] << std::endl;
    // std::cout << "position_z: " << tvec.ptr<double>(0)[2] << std::endl;
  }

  // 发布检测结果
  pub.publish(points);
}
