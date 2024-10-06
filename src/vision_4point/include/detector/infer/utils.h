#ifndef TJURM_INCLUDE_DETECTOR_INFER_UTILS_H_
#define TJURM_INCLUDE_DETECTOR_INFER_UTILS_H_

#include <opencv2/opencv.hpp>
#include <vector>

namespace detector {
// 没有什么特别大的用处，仅用于计算model输出所占的大小
struct alignas(float) Detection0 {
  float bbox[8];               // center_x, center_y, w, h
  float conf;                  // bbox_conf * cls_conf
  float class_possibility[3]; // number of classes depend on model
};

struct alignas(float) Detection {
  float bbox[8]; // center_x, center_y, w, h
  float conf;    // bbox_conf * cls_conf
  float class_id;
};

// 描述一个目标
struct Object {
  float confidence;     // 置信度
  cv::Rect box;         // 矩形框
  cv::Point2f point[4]; // 装甲板四个顶点
  int id;               // 类别
};

// 描述一个tensor的形状
struct Shape {
  int n, h, w, c;
};

namespace utils {

// 把一个tensor的内存顺序从N H W C转化为N C H W
// 因为opencv的内存顺序的N H W C，而pytorch用的是N C H W
void nhwc_to_nchw(float *src, float *dst, Shape shape);

// 找出长度为n的数组a中最大值所在的位置（下标）
int argmax(float *a, int n);

// x = tanh(x)
void tanh(float *a, int n);

// 找出长度为n的数组a中最小值，可以设置起点与步长
float min(float *a, int n, int start, int s);

// 找出长度为n的数组a中最大值，可以设置起点与步长
float max(float *a, int n, int start, int s);

bool compareDetection(const Detection &a, const Detection &b);

// 计算IOU
float iou(float a[4], float b[4]);
float ComputeIOU(cv::Rect box1, cv::Rect box2);

// 输出的数据(640*640)还需要恢复成原图的尺寸标准
void get_rect_point(int cols, int rows, float box[8], int kInputW, int kInputH,
                    cv::Rect &rect, cv::Point2f point[4]);

// 极大值抑制
std::vector<std::vector<Object>> NoneMaxSupression(std::vector<Object> &objs,
                                                   float max_iou);
std::vector<Object> NoneMaxSupression_new(std::vector<Object> &objs,
                                          float max_iou);
void nms(std::vector<std::vector<Object>> &res, float *output,
         int kMaxNumOutputBbox, float conf_thresh, float nms_thresh,
         int kInputW, int kInputH, int cols, int rows);

// 将网络的输出进行解码
std::vector<Object> decode(float *output, float conf_thresh);

void ShowObjects(cv::Mat &img, std::vector<Object> &objs);

void ShowPoints(cv::Mat &img, std::vector<Object> &objs);

} // namespace utils

} // namespace detector

#endif // TJURM_INCLUDE_DETECTOR_INFER_UTILS_H_
