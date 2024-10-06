#ifndef TJURM_INCLUDE_DETECTOR_INFER_TRT_H_
#define TJURM_INCLUDE_DETECTOR_INFER_TRT_H_


#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include "utils.h"
#include "infer.h"

#include "common.h"

using namespace nvinfer1;
using namespace nvonnxparser;

namespace detector
{

/* only support neural network with one input tensor */
class TRT : public Detector {
private:
    /* input and output tensors in cpu and gpu */
    /* can be indexed by their names */
    cudaStream_t       stream_;

    int                kInputW_;
    int                kInputH_;
    int                kOutputSize_;
    float              kConfThresh_;
    float              kNmsThresh_;
    int                kMaxNumOutputBbox_;

    uint8_t*           img_buffer_host_;    // cpu中分配的一块固定内存(pinned memory)地址
    uint8_t*           img_buffer_device_;  // cpu中分配的一块内存

    /* tensorRT engine and context */
    IRuntime*          runtime_;
    ICudaEngine*       engine_;
    IExecutionContext* context_;

public:
    TRT();
    ~TRT();

public:
    void load_from_onnx(const std::string& onnx_path);
    void load_from_cache(const std::string& cache_path);
    // core dumped ?
    void save_engine(const std::string& cache_path);
    void bind();

public:
    void init();
    std::vector<std::vector<Object>> detect(const cv::Mat& src);

private:
    void prepare_buffers(float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
    void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height);
    void infer(void** gpu_buffers, float* output, int batchsize);

};

} // namespace detector


#endif // TJURM_INCLUDE_DETECTOR_INFER_TRT_H_
