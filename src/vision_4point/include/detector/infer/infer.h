#ifndef TJURM_INCLUDE_DETECTOR_INFER_INFER_H_
#define TJURM_INCLUDE_DETECTOR_INFER_INFER_H_

// #include <opencv2/opencv.hpp>
#include "utils.h"
#include "common.h"


namespace detector
{

/* 目标检测网络的基类 */
class Detector {
public:
    Detector() = default;
    ~Detector() = default;
public:
    virtual void init() = 0;
    virtual std::vector<std::vector<Object>> detect(const cv::Mat& src) = 0;
};

} // namespace detector

#endif // TJURM_INCLUDE_DETECTOR_INFER_INFER_H_
