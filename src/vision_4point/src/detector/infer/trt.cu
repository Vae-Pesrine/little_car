#include "detector/infer/trt.h"
#include <unistd.h>

using std::vector;

struct AffineMatrix {
  float value[6];
};

// 定义的一个cuda核函数，这个核函数的用处简单来讲，实现的就是resize的功能
// 但相比于opencv的resize，用cuda核函数可以实现并行处理，速度快很多
__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge) return;

  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  int dx = position % dst_width;
  int dy = position / dst_width;
  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t* v1 = const_value;
    uint8_t* v2 = const_value;
    uint8_t* v3 = const_value;
    uint8_t* v4 = const_value;

    if (y_low >= 0) {
      if (x_low >= 0)
        v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width)
        v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0)
        v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width)
        v4 = src + y_high * src_line_size + x_high * 3;
    }

    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // bgr to rgb
  float t = c2;
  c2 = c0;
  c0 = t;

  // normalization
  // printf("c0: %f, c1: %f, c2: %f\n", c0, c1, c2);
  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  // rgbrgbrgb to rrrgggbbb
  int area = dst_width * dst_height;
  float* pdst_c0 = dst + dy * dst_width + dx;
  float* pdst_c1 = pdst_c0 + area;
  float* pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

/* convert dims to 1-d size */
static int dims_to_size(Dims dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++)
        size *= dims.d[i];
    return size;
}

detector::TRT::TRT()
{}

detector::TRT::~TRT() {
    cudaStreamDestroy(this->stream_);

    cudaFree(this->img_buffer_device_);
    cudaFreeHost(this->img_buffer_host_);

    this->runtime_->destroy();
    this->context_->destroy();
    this->engine_->destroy();
}


void detector::TRT::load_from_onnx(const std::string& onnx_path) {
    std::cout << "loading from onnx ..." << std::endl;;
    /* create a logger */
    Logger logger;

    /* create a builder for building and optimizing the network */
    auto builder = createInferBuilder(logger);

    /* create network */
    const auto explicitBatch =
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    /* create parser */
    auto parser = nvonnxparser::createParser(*network, logger);

    /* parse onnx and store informaion and weights to network */
    parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kINFO));

    /* set names for input and output tensors */
    //TensorRt: 2.网络输出要改！
    network->getInput(0)->setName("input");
    network->getOutput(0)->setName("output");
    // network->getOutput(1)->setName("output_11x11x15");

    /* create config for builder */
    auto config = builder->createBuilderConfig();

    /* check fp16 */
    if (builder->platformHasFastFp16())
    {
        std::cout << "platform support fp16, enable fp16" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    } else {
        std::cout << "platform do not support fp16, enable fp32" << std::endl;
    }

    /* check GPU information */
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "total gpu mem: " << (total >> 20)
              << "MB, free gpu mem: " << (free >> 20)
              << "MB" << std::endl;
    std::cout << "max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(free);

    /* create engine and context for executing the network */
    engine_ = builder->buildEngineWithConfig(*network, *config);

    context_ = engine_->createExecutionContext();

    /* release everything */
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
}

void detector::TRT::load_from_cache(const std::string& cache_path) {
    // MSG("loading from cache: {}", cache_path);
    std::cout << "loading from cache ..." << std::endl;;

    Logger logger;

    // deserialize engine
    std::ifstream file(cache_path, std::ios::binary);    // 以二进制形式加载engine
    size_t size = 0;
    file.seekg(0, file.end);    // 将读指针从文件末尾开始移动0个字节
    size = file.tellg();    // 返回读指针的位置，此时读指针的位置就是文件的字节数
    file.seekg(0, file.beg);    // 将读指针从文件开头开始移动0个字节
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);   // 将文件的内容读入serialized_engine
    file.close();

    this->runtime_ = createInferRuntime(logger);
    this->engine_ = this->runtime_->deserializeCudaEngine(serialized_engine, size);   // 将serialized_engine中的内容反序列化为一个ICudaEngine对象，并将其地址存储在engine中
    this->context_ = engine_->createExecutionContext();

    delete[] serialized_engine;
}

void detector::TRT::save_engine(const std::string& cache_path) {
    /* serilalize the engine */
    auto engine_buffer = engine_->serialize();

    std::ofstream ofs(cache_path, std::ios::binary);
    ofs.write(static_cast<const char *>(engine_buffer->data()), engine_buffer->size());

    engine_buffer->destroy();
}

void detector::TRT::bind() {
    /* initialize cuda */
    // MSG("initialize cuda");
    cudaSetDevice(0);
    cudaStreamCreate(&stream_);

    // If your image size is larger than 4096 * 3112, please increase this value
    int max_image_size = 4096 * 3112;
    // prepare input data in pinned memory
    // MSG("prepare input data in pinned memory");
    cudaMallocHost((void**)&img_buffer_host_, max_image_size * 3); // cpu中分配一块固定内存(pinned memory)，并将其地址存储在全局变量img_buffer_host中,可以被GPU直接访问，从而提高数据传输的效率
    // prepare input data in device memory
    cudaMalloc((void**)&img_buffer_device_, max_image_size * 3);   // cpu中分配一块内存，并将其地址存储在全局变量img_buffer_device中
}

void detector::TRT::init() {
    this->kInputW_ = 640;
    this->kInputH_ = 640;
    this->kConfThresh_ = 0.5f;
    this->kNmsThresh_ = 0.30f;
    this->kMaxNumOutputBbox_ = 25200;
    this->kOutputSize_ = kMaxNumOutputBbox_ * sizeof(Detection0) / sizeof(float) + 1;

    this->engine_ = nullptr;
    this->context_ = nullptr;
    this->img_buffer_host_ = nullptr;
    this->img_buffer_device_ = nullptr;

    if (access("/home/tjurm/Desktop/int_car/src/vision_4point/assets/circle.engine", F_OK) == 0) {
        this->load_from_cache("/home/tjurm/Desktop/int_car/src/vision_4point/assets/circle.engine");
    } else {
        this->load_from_onnx("/home/tjurm/Desktop/int_car/src/vision_4point/assets/circle.onnx");
        this->save_engine("/home/tjurm/Desktop/int_car/src/vision_4point/assets/circle.engine");
    }
    this->bind();
}

void detector::TRT::prepare_buffers(float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
    // assert(engine_->getNbBindings() == 2);   // 检查engine绑定的输入和输出张量的数量是否正确

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine_->getBindingIndex("images");   // kInputTensorName是输入张量的名称
    // const int outputIndex = engine_->getBindingIndex("output0"); // kOutputTensorName是输出张量的名称
    // assert(inputIndex == 0);
    // assert(outputIndex == 1);

    // Create GPU buffers on device
    cudaMalloc((void**)gpu_input_buffer, 1 * 3 * this->kInputH_ * this->kInputW_ * sizeof(float));
    cudaMalloc((void**)gpu_output_buffer, 1 * this->kOutputSize_ * sizeof(float));

    *cpu_output_buffer = new float[1 * kOutputSize_]; // 创建一个大小为 kBatchSize * kOutputSize 的浮点数数组，并将其地址赋值给*cpu_output_buffer。这个数组可以用来存储模型的输出结果。
}

void detector::TRT::cuda_preprocess(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height) {
    int img_size = src_width * src_height * 3;
    // copy data to pinned memory
    memcpy(this->img_buffer_host_, src, img_size);   // 将输入图像数据复制到pinned内存中，img_buffer_host是指向设置好pinned内存的指针全局变量
    // copy data to device memory
    cudaMemcpyAsync(this->img_buffer_device_, this->img_buffer_host_, img_size, cudaMemcpyHostToDevice, this->stream_);  // 在stream流中，将pinned内存的数据异步复制到设备内存中

    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);  // 2x3 的矩阵，存储了仿射变换矩阵 s2d 的值
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);  // 空的 Mat 对象m2x3_d2s，用于存储逆变换矩阵
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);  // 将s2d的逆矩阵计算出来并存储在d2s矩阵中，在进行反向变换时，就可以使用d2s矩阵来进行变换操作。

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    // 在stream_流中执行名为warpaffine_kernel的CUDA核函数
    warpaffine_kernel<<<blocks, threads, 0, this->stream_>>>(  // 启动了blocks个线程块，每个线程块包含threads个线程，共计blocks * threads个线程，共享内存0字节,并指定在stream流中启动核函数
        this->img_buffer_device_, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 114, d2s, jobs);
}

void detector::TRT::infer(void** gpu_buffers, float* output, int batchsize) {
    // this->context_->enqueue(batchsize, gpu_buffers, this->stream_, nullptr);   // 将批处理大小个输入数据放入GPU缓冲区中，并在指定的CUDA流上异步执行推理
    this->context_->enqueueV2(gpu_buffers, this->stream_, nullptr);
    cudaMemcpyAsync(
        output, gpu_buffers[1],             // 异步地将GPU缓冲区gpu_buffers[1]的数据复制到主机内存中的output地址；
        batchsize * this->kOutputSize_ * sizeof(float),     // 第三个参数是要复制的数据大小，以字节为单位；
        cudaMemcpyDeviceToHost, this->stream_);     // 第四个参数是复制的方向，这里是从设备(GPU)到主机(Host)；第五个参数是指定的CUDA流，表示在哪个流上执行复制操作

    cudaStreamSynchronize(this->stream_);    // 等待CUDA流上的所有操作完成，以确保输出数据已经准备好
}

vector<vector<detector::Object>> detector::TRT::detect(const cv::Mat& src) {
    // Prepare cpu and gpu buffers
    // MSG("Prepare cpu and gpu buffers");
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    // MSG("this->prepare_buffers(&gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);");
    this->prepare_buffers(&gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    // Preprocess
    // MSG("Preprocess");
    //FIXME: 严重bug！深拷贝浅拷贝
    cv::Mat img = src;      // 没有这段代码的话下面的函数可能会修改src，导致报错
    this->cuda_preprocess(img.ptr(), src.cols, src.rows, gpu_buffers[0], this->kInputW_, this->kInputH_);
    cudaStreamSynchronize(this->stream_);

    // systime start1; GetSystime(start1);
    // Run inference
    // MSG("Run inference");
    this->infer((void**)gpu_buffers, cpu_output_buffer, 1);
    // MSG("here");
    // systime end1; GetSystime(end1);
    // MSG("time5 = {:.3f} ms", (end1 - start1));
    

    /* non-max supression operation */
    // 由于当前网络输出多个类别的检测结果，因此需要对每个类别的检测结果进行NMS操作，这样的话直接就返回最终结果了！
    vector<vector<Object>> objs;
    utils::nms(objs, cpu_output_buffer, this->kMaxNumOutputBbox_, this->kConfThresh_, this->kNmsThresh_, this->kInputW_, this->kInputH_, src.cols, src.rows);

    // Detection====>Object
    // vector<Object> cars;
    // vector<Object> armors;
    // for (auto it = res.begin(); it != res.end(); it++) {
    //     Object obj;
    //     obj.confidence = it->conf;
    //     obj.box = utils::get_rect(img, it->bbox, this->kInputW_, this->kInputH_);
    //     obj.id = it->class_id;
    //     // objects.push_back(obj);
    //     if(obj.id == 0)
    //         cars.push_back(obj);
    //     else
    //         armors.push_back(obj);
    // }

    cudaFree(gpu_buffers[0]);
    cudaFree(gpu_buffers[1]);
    delete[] cpu_output_buffer;
    return objs;
}
