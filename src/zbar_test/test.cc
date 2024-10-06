#include "iostream"
#include "opencv2/opencv.hpp"
#include "zbar.h"

using namespace std;
using namespace cv;
using namespace zbar;

#define WINDOW_NAME "clor"
#define WINDOW_GARY_NAME "gary"

int main() {
  namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
  namedWindow(WINDOW_GARY_NAME, WINDOW_AUTOSIZE);

  // 加载二维码图片
  Mat image;
  // image = imread("./picture/test.jpg");
  image = imread("../picture/1.png");
  // image = imread("./picture/barCode.jpg");
  // image = imread("./picture/2Code.png");
  // image = Mat(240,320,CV_8UC3,Scalar(0,255,0));

  cout << "---------------- 图像参数 ------------------" << endl;
  // 标志位
  cout << "flags:" << image.flags << endl;
  // 图像尺寸
  cout << "size:" << image.size << endl;
  // 列宽
  cout << "clos:" << image.cols << endl;
  // 行高
  cout << "rows:" << image.rows << endl;
  // 维度
  cout << "dims:" << image.dims << endl;

  cout << "------------------------------------------" << endl;

  imshow(WINDOW_NAME, image);

  // 灰度转换
  Mat imageGray;
  cvtColor(image, imageGray, COLOR_RGB2GRAY);
  imshow(WINDOW_GARY_NAME, imageGray);

  // 获取二进制数据
  int width = imageGray.cols;
  int height = imageGray.rows;
  uchar *raw = (uchar *)imageGray.data;
  Image imageZbar = Image(width, height, "Y800", raw, width * height);

  // 配置扫描器,开始扫描
  ImageScanner scanner;
  scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
  scanner.scan(imageZbar);

  // 扫描结果打印
  if (imageZbar.symbol_begin() == imageZbar.symbol_end()) {
    cout << "识别错误！" << endl;
  }
  // 遍历所有识别到的二维码后者条形码
  Image::SymbolIterator symbol = imageZbar.get_symbols();
  for (; symbol != imageZbar.symbol_end(); ++symbol) {
    cout << "类型：\t" << symbol->get_type_name() << endl;
    cout << "条码：\t" << symbol->get_data() << endl << endl;
  }
  // 释放资源
  imageZbar.set_data(NULL, 0);

  waitKey(0);

  return 0;
}
