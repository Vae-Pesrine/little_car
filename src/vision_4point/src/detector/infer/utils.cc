#include "detector/infer/infer.h"
#include <cmath>

using detector::Object;
using std::vector;

// 计算位置为(b, z, y, x)的元素在内存顺序为N-C-H-W的tensor中的索引
// b是图片在batch中的下标，z是通道方向的下标，y是高度方向的下标，x是宽度方向的下标
#define index_nchw(c, h, w, b, z, y, x) (b * c * h * w + z * h * w + y * w + x)

// 计算位置为(b, z, y, x)的像素在内存顺序为N-H-W-C的图片中的索引
// b是图片在batch中的下标，z是通道方向的下标，y是高度方向的下标，x是宽度方向的下标
#define index_nhwc(c, h, w, b, z, y, x) (b * h * w * c + y * w * c + x * c + z)

void detector::utils::nhwc_to_nchw(float *src, float *dst, Shape shape) {
  int n = shape.n, h = shape.h, w = shape.w, c = shape.c;
  for (int b = 0; b < n; b++) {
// 使用openmp将下面的循环并行，效果是一样的，只不过速度更快了
#pragma omp parallel for num_threads(2) // two two two two two
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        for (int z = 0; z < c; z++) {
          // 找到对应的位置，拷贝一下就行
          int src_index = index_nhwc(c, h, w, b, z, y, x);
          int dst_index = index_nchw(c, h, w, b, z, y, x);
          dst[dst_index] = src[src_index];
        }
  }
}

int detector::utils::argmax(float *a, int n) {
  // 太简单了，不讲
  int res = 0;
  float max = a[res];
  for (int i = 1; i < n; i++)
    if (a[i] > max) {
      max = a[i];
      res = i;
    }
  return res;
}

void detector::utils::tanh(float *a, int n) {
  // 太简单了，不讲
  for (int i = 0; i < n; ++i)
    a[i] = (exp(a[i]) - exp(-a[i])) / (exp(a[i]) + exp(-a[i]));
}

// FIXME: 大bug，即然都能设置start了，那么第一个显然有可能不是我们想进入比较的！
float detector::utils::min(float *a, int n, int start, int stride) {
  // 太简单了，不讲
  float res = a[0];
  for (int i = start; i < n; i += stride)
    res = (a[i] < res) ? a[i] : res;
  return res;
}

float detector::utils::max(float *a, int n, int start, int stride) {
  // 太简单了，不讲
  float res = a[0];
  for (int i = start; i < n; i += stride)
    res = (a[i] > res) ? a[i] : res;
  return res;
}

bool detector::utils::compareDetection(const Detection &a, const Detection &b) {
  return a.conf > b.conf;
}

float detector::utils::iou(float a[4], float b[4]) {
  float x1 = std::max(a[0] - a[2] / 2.f, b[0] - b[2] / 2.f);
  float y1 = std::max(a[1] - a[3] / 2.f, b[1] - b[3] / 2.f);
  float x2 = std::min(a[0] + a[2] / 2.f, b[0] + b[2] / 2.f);
  float y2 = std::min(a[1] + a[3] / 2.f, b[1] + b[3] / 2.f);

  float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  float area_a = a[2] * a[3];
  float area_b = b[2] * b[3];
  float union_area = area_a + area_b - intersection;

  return intersection / union_area;
}

float detector::utils::ComputeIOU(cv::Rect box1, cv::Rect box2) {
  /*
  ----------------------------------> x
  |    A----------
  |    |         |
  |    |    B------------
  |    |    ||||||      |
  |    -----|----C      |
  |         |           |
  |         ------------D
  y

  A坐标：(box1.x, box1.y)
  B坐标：(box2.x, box2.y)
  相交区域左上角坐标：(max(box1.x, box2.x), max(box1.y, box2.y))
  右下角坐标同理
   */

  // 计算重叠区域左上角坐标
  int x1 = std::max(box1.x, box2.x);
  int y1 = std::max(box1.y, box2.y);
  // 计算重叠区域右下角坐标
  int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
  int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
  // 计算重叠区域宽高
  int w = std::max(0, x2 - x1 + 1);
  int h = std::max(0, y2 - y1 + 1);
  float over_area = w * h;
  return over_area / (box1.width * box1.height + box2.width * box2.height -
                      over_area + 1e-5 /* 防止除以0 */);
}

vector<vector<Object>> detector::utils::NoneMaxSupression(vector<Object> &objs,
                                                          float max_iou) {
  // 存储过滤后的结果
  vector<Object> res;

  // 根据置信度对所有的候选框进行排序
  std::sort(objs.begin(), objs.end(),
            // lambda表达式可以看做是一个函数，作为sort函数的参数
            [](Object &a, Object &b) { return a.confidence > b.confidence; });
  // std::cout << "number of objects before nms: " << objs.size() << std::endl;

  // 还存在候选框时
  while (objs.size() > 0) {
    // 留下第一个候选框
    res.push_back(objs.at(0));

    // 检查后面所有的候选框
    int index = 1;
    while (index < objs.size()) {
      // 计算一下这个候选框跟刚刚留下的候选框的IOU
      float iou = ComputeIOU(objs.at(0).box, objs.at(index).box);
      // 超过阈值，就把它从vector中清除
      if (iou > max_iou)
        objs.erase(objs.begin() + index);
      else
        index++;
    }

    // 取走留下的候选框
    objs.erase(objs.begin());
  }
  // std::cout << "number of objects after nms: " << res.size() << std::endl;
  vector<vector<Object>> results;
  results.push_back(res);
  results.push_back(res);
  return results;
}

vector<Object> detector::utils::NoneMaxSupression_new(vector<Object> &objs,
                                                      float max_iou) {
  // 存储过滤后的结果
  vector<Object> res;

  // 根据置信度对所有的候选框进行排序
  std::sort(objs.begin(), objs.end(),
            // lambda表达式可以看做是一个函数，作为sort函数的参数
            [](Object &a, Object &b) { return a.confidence > b.confidence; });
  // std::cout << "number of objects before nms: " << objs.size() << std::endl;

  // 还存在候选框时
  while (objs.size() > 0) {
    // 留下第一个候选框
    res.push_back(objs.at(0));

    // 检查后面所有的候选框
    int index = 1;
    while (index < objs.size()) {
      // 计算一下这个候选框跟刚刚留下的候选框的IOU
      float iou = ComputeIOU(objs.at(0).box, objs.at(index).box);
      // 超过阈值，就把它从vector中清除
      if (iou > max_iou)
        objs.erase(objs.begin() + index);
      else
        index++;
    }

    // 取走留下的候选框
    objs.erase(objs.begin());
  }
  // std::cout << "number of objects after nms: " << res.size() << std::endl;
  return res;
}

void detector::utils::get_rect_point(int cols, int rows, float box[8],
                                     int kInputW, int kInputH, cv::Rect &rect,
                                     cv::Point2f point[4]) {
  // box[8] = {x1,y1,x2,y2,x3,y3,x4,y4}
  // bbox[4] = {x,y,w,h}
  // box[8]  求外接矩形 得到bbox[4]
  float x_min = min(box, 8, 0, 2);
  float x_max = max(box, 8, 0, 2);
  float y_min = min(box + 1, 7, 0, 2);
  float y_max = max(box + 1, 7, 0, 2);
  float bbox[4] = {(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min,
                   y_max - y_min};
  float x1, y1, x2, y2, x3, y3, x4, y4;

  float l, r, t, b;
  float r_w = kInputW / (cols * 1.0);
  float r_h = kInputH / (rows * 1.0);
  if (r_h > r_w) {
    l = bbox[0] - bbox[2] / 2.f;
    r = bbox[0] + bbox[2] / 2.f;
    t = bbox[1] - bbox[3] / 2.f - (kInputH - r_w * rows) / 2;
    b = bbox[1] + bbox[3] / 2.f - (kInputH - r_w * rows) / 2;
    x1 = box[0] / r_w;
    y1 = (box[1] - (kInputH - r_w * rows) / 2) / r_w;
    x2 = box[2] / r_w;
    y2 = (box[3] - (kInputH - r_w * rows) / 2) / r_w;
    x3 = box[4] / r_w;
    y3 = (box[5] - (kInputH - r_w * rows) / 2) / r_w;
    x4 = box[6] / r_w;
    y4 = (box[7] - (kInputH - r_w * rows) / 2) / r_w;
    l = l / r_w;
    r = r / r_w;
    t = t / r_w;
    b = b / r_w;
  } else {
    l = bbox[0] - bbox[2] / 2.f - (kInputW - r_h * cols) / 2;
    r = bbox[0] + bbox[2] / 2.f - (kInputW - r_h * cols) / 2;
    t = bbox[1] - bbox[3] / 2.f;
    b = bbox[1] + bbox[3] / 2.f;
    x1 = (box[0] - (kInputW - r_h * cols) / 2) / r_h;
    y1 = box[1] / r_h;
    x2 = (box[2] - (kInputW - r_h * cols) / 2) / r_h;
    y2 = box[3] / r_h;
    x3 = (box[4] - (kInputW - r_h * cols) / 2) / r_h;
    y3 = box[5] / r_h;
    x4 = (box[6] - (kInputW - r_h * cols) / 2) / r_h;
    y4 = box[7] / r_h;
    l = l / r_h;
    r = r / r_h;
    t = t / r_h;
    b = b / r_h;
  }
  rect = cv::Rect(round(l), round(t), round(r - l), round(b - t));
  point[0] = cv::Point2f(x1, y1);
  point[1] = cv::Point2f(x2, y2);
  point[2] = cv::Point2f(x3, y3);
  point[3] = cv::Point2f(x4, y4);
}

void detector::utils::nms(std::vector<std::vector<Object>> &res, float *output,
                          int kMaxNumOutputBbox, float conf_thresh,
                          float nms_thresh, int kInputW, int kInputH, int cols,
                          int rows) {

  // Detection0是网络的原始输出，而Detection是分析出类别之后的
  int det_size0 = sizeof(Detection0) / sizeof(float);
  int det_size = sizeof(Detection) / sizeof(float);
  std::vector<Object> armors;
  for (int i = 0; i < kMaxNumOutputBbox; i++) {
    //首先滤掉所有置信度较低的框
    if (output[det_size0 * i + 8] <= conf_thresh) {
      continue;
    }
    //找到置信度最高的类别
    int max_index = 0;
    double max_possibility = output[det_size0 * i + 9];
    for (int j = 1; j < det_size0 - 9; j++) {
      if (output[det_size0 * i + 9 + j] > max_possibility) {
        max_possibility = output[det_size0 * i + 9 + j];
        max_index = j;
      }
    }
    //这里将Detection0两个类的置信度直接嘎掉，替换成最终的类别
    Detection det;
    memcpy(&det, &output[det_size0 * i], det_size * sizeof(float));
    det.class_id = max_index;
    // 注意上面得到的Detection并不是我们最终需要的目标信息，我们还需要将其转换我们定义的Object对象
    Object obj;
    obj.confidence = det.conf;
    get_rect_point(cols, rows, det.bbox, kInputW, kInputH, obj.box, obj.point);
    obj.id = det.class_id;
    // if(obj.id == 1)
    //     armors.push_back(obj);
    // else
    //     cars.push_back(obj);
    armors.push_back(obj);
  }
  // cars = NoneMaxSupression_new(cars, nms_thresh);
  armors = NoneMaxSupression_new(armors, nms_thresh);
  // res.push_back(cars);
  res.push_back(armors);
  //根据置信度对所有的候选框进行排序
  // sort(res.begin(), res.end(), compareDetection);
  // for (auto it1 = res.begin(); it1 != res.end(); it1++) {
  //     auto it2 = it1;
  //     it2++;
  //     while (it2 != res.end()) {
  //         if (iou(it1->bbox, it2->bbox) > nms_thresh) {
  //             res.erase(it2);
  //             continue;
  //         }
  //         it2++;
  //     }
  // }
}

// memory order of output tensor is H-W-C
// this function do not deal with mutiple samples
// ori_h, ori_w: 原始图像的大小（输入网络前）
// input_h, input_w: 网络的输入图像大小
// output: 网络的输出结果
// feat_mat_h, feat_map_w: 输出特征图的大小
// channels: 网络的输出结果的通道数
// n_anchor: anchor的数量
// conf_thresh: 置信度阈值

// TensorRt: 7.decode cc
vector<Object> detector::utils::decode(float *output, float conf_thresh) {
  vector<Object> objs;
  int tmp_idx = 0;
  for (int i = 0; i < 25200; i++) {
    Object obj;
    tmp_idx = i * (1 + 5);
    float w = output[tmp_idx + 2];
    float h = output[tmp_idx + 3] * 0.75;
    float x =
        output[tmp_idx + 0] -
        output[tmp_idx + 2] /
            2; // 从第一个指针开始，每（cls_num+5）个数据中，前4位对应x,y,w,h
    float y = (output[tmp_idx + 1] - output[tmp_idx + 3] / 2) * 0.75;
    obj.box = cv::Rect(x, y, w, h);
    if (output[tmp_idx + 4] < conf_thresh) // filiter
      continue;
    obj.confidence = output[tmp_idx + 4]; // 是为目标的置信度
    // obj.id = 0;
    objs.push_back(obj);
  }
  // // 计算缩放系数
  // float fx = (float)ori_w / input_w, fy = (float)ori_h / input_h;
  // // 计算特征图上的每一个网格对应的网络输入图像的大小
  // float stride_w = (float)input_w / feat_map_w, stride_h = (float)input_h /
  // feat_map_h;

  // for (int i = 0; i < feat_map_h; i++)
  //     for (int j = 0; j < feat_map_w; j++) {
  //         // 网格(i, j)对应的输出向量
  //         // 向量各个分量的含义：
  //         // |----------------|-------------------------------------|
  //         // |<--confidence-->|<--            box                -->|
  //         // 各个anchor的置信度           各个anchor的矩形框
  //         float* v = output + i * feat_map_w * channels + j * channels;

  //         // 置信度的起点
  //         float* conf_seg = v;
  //         // 矩形框的起点
  //         float* box_seg = v + 1 * n_anchor;

  //         // 遍历每一个anchor的结果
  //         for (int a = 0; a < n_anchor; a++) {
  //             // 置信度
  //             float conf = conf_seg[a];
  //             // 矩形框
  //             float* box = box_seg + 4 * a;
  //             // 根据公式对网络输出进行解码，参考yolov5
  //             // x, y

  //             float cx = (box[0] * 2 - 0.5 + j) * stride_w * fx,
  //                   cy = (box[1] * 2 - 0.5 + i) * stride_h * fy;
  //             // w, h
  //             float* anchor = anchors + a * 2;
  //             float bw = pow(box[2] * 2, 2) * anchor[0] * fx;
  //             float bh = pow(box[3] * 2, 2) * anchor[1] * fy;

  //             // 根据置信度阈值进行过滤
  //             if (conf > conf_thresh) {
  //                 Object o;
  //                 o.confidence = conf;
  //                 o.box = cv::Rect(cx - bw / 2, cy - bh / 2, bw, bh);
  //                 objs.push_back(o);
  //             }
  //         }
  //     }
  return objs;
}

void detector::utils::ShowObjects(cv::Mat &img, vector<Object> &objs) {
  for (auto obj : objs) {
    // std::cout << obj.confidence << " " << obj.id << " "
    //           << obj.box[0] << " " << obj.box[1] << " "
    //           << obj.box[2] << " " << obj.box[3] << std::endl;
    cv::rectangle(img, obj.box, cv::Scalar(255, 255, 0));
  }
}

void detector::utils::ShowPoints(cv::Mat &img, vector<Object> &objs) {
  for (auto obj : objs) {
    // std::cout << obj.confidence << " " << obj.id << " "
    //           << obj.box[0] << " " << obj.box[1] << " "
    //           << obj.box[2] << " " << obj.box[3] << std::endl;
    cv::line(img, obj.point[0], obj.point[1], cv::Scalar(255, 255, 0), 2);
    cv::line(img, obj.point[1], obj.point[2], cv::Scalar(255, 255, 0), 2);
    cv::line(img, obj.point[2], obj.point[3], cv::Scalar(255, 255, 0), 2);
    cv::line(img, obj.point[3], obj.point[0], cv::Scalar(255, 255, 0), 2);
  }
}
