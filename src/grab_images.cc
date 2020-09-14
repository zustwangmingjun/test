#include <fcntl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

#include <ctime>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "convert.hpp"

#define WIDTH 640   // max: depth 1280; color 1920
#define HEIGHT 480  // min:        720;       1080
#define FRAMERATE 30
#define IMAGE_NUM 15

using namespace std;
using namespace cv;

// 函数是对深度图二值化，第一个参数image是原图，第二个参数th是目标图，第三个参数throld是最大距离，单位是mm，
// 大于这个距离,即为安全，不用考虑。
void mask_depth(Mat &image, Mat &th, int throld = 1000) {
  int nr = image.rows;  // number of rows 行数
  int nc = image.cols;  // number of columns 列数
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nc; j++) {
      // if (image.at<ushort>(i, j) > throld || i > 455 || j < 160 || j > 480)
      if (image.at<ushort>(i, j) > throld) th.at<ushort>(i, j) = 0;
    }
  }
}

// 获取深度图障碍物的函数，返回值是每个障碍物凸包的坐标
// 参数一depth是realsense返回的深度图（ushort型）
// 参数二thresh和参数三max_thresh，是二值化的参数
// 参数四是凸包的最小有效面积，小于这个面积的障碍物可以视为噪点
vector<vector<Point>> find_obstacle(Mat &depth, int thresh = 15,
                                    int max_thresh = 255, int area = 500) {
  Mat dep;

  // imshow("depth", depth);

  // 得到和depth一样的矩阵dep 不一定申请新的内存空间
  depth.copyTo(dep);
  cout << "head_dep" << endl;
  cout << dep << endl;

  // 将安全距离外的像素点置零 不考虑
  mask_depth(depth, dep, 4500);

  dep.convertTo(dep, CV_8UC1, 1.0 / 4.0);

  // cout << "dep_8" << endl;
  // cout << dep << endl;
  // dep.convertTo(dep, CV_8UC1, 1.0 / 16);

  Mat element =
      getStructuringElement(MORPH_RECT, Size(15, 15));  //核的大小可适当调整
  Mat out;
  // 进行形态学操作 开操作
  // morphologyEx(dep, out, MORPH_OPEN, element);

  // 闭操作
  morphologyEx(dep, out, MORPH_CLOSE, element);
  // dilate(dhc, out, element);

  // 显示处理后的效果图
  // imshow("opencv", out);

  // 是完全的深拷贝，在内存中申请新的空间，src_copy与dep独立
  Mat src_copy = dep.clone();
  Mat threshold_output;

  // 是一个向量，并且是一个双重向量，向量内每个元素保存了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓。
  vector<vector<Point>> contours;

  // 是一个向量，向量内每个元素保存了一个包含4个int整型的数组
  // 向量hierarchy内的元素和轮廓向量contours内的元素是一一对应的，向量的容量相同。
  // hierarchy向量内每一个元素的4个int型变量——hierarchy[i][0]
  // ~hierarchy[i][3]，分别表示第i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号。
  vector<Vec4i> hierarchy;

  // 将状态设定为指定值
  RNG rng(12345);

  // 对图像进行二值化
  threshold(dep, threshold_output, thresh, 255, CV_THRESH_BINARY);
  // mask_depth(src, threshold_output);
  cout << "head_binary" << endl;
  // cout << threshold_output << endl;

  imshow("threshold_output", threshold_output);

  // 轮廓检索模式
  // CV_RETR_EXTERNAL:只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
  // CV_RETR_LIST:检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立，没有等级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓
  // CV_RETR_CCOMP:检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层
  // CV_RETR_TREE:
  // 检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。
  // CV_RETR_FLOODFILL
  // 轮廓近似方法
  // CV_CHAIN_APPROX_NONE:保存物体边界上所有连续的轮廓点到contours向量内
  // CV_CHAIN_APPROX_SIMPLE:仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留
  // CV_CHAIN_APPROX_TC89_L1 和 CV_CHAIN_APPROX_TC89_KCOS
  findContours(
      threshold_output,  // 单通道图像矩阵，可以是灰度图，通常是二值图，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像；
      contours,          //双重向量
      hierarchy,         //轮廓的索引编号
      CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));  //偏移量

  // 对每个轮廓计算其凸包
  vector<vector<Point>> hull(contours.size());
  vector<vector<Point>> result;

  // cout << "size1  " << contours.size() << endl;

  // 得到每个障碍物的凸包
  for (uint i = 0; i < contours.size(); i++) {
    convexHull(Mat(contours[i]),  // 输入的二维点集，Mat类型数据即可
               hull[i],  // 输出参数，用于输出函数调用后找到的凸包
               false);  // 操作方向逆时针
  }

  // 绘出轮廓及其凸包
  Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
  for (uint i = 0; i < contours.size(); i++) {
    // cout << contourArea(contours[i]) << endl;
    if (contourArea(contours[i]) < area)  // 面积小于area的凸包，可忽略
      continue;
    result.push_back(hull[i]);

    // 3个参数分别为BGR
    Scalar color =
        Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    drawContours(drawing, contours, i, color, 3, 8, vector<Vec4i>(), 0,
                 Point());
    drawContours(drawing,  // 目标图像
                 hull,     // 输入轮廓组、凸包组
                 i,        // 画第几个轮廓
                 color,    // 轮廓颜色
                 3,  // 轮廓线的粗细，如果为负值或CV_FILLED表示填充轮廓内部
                 8,                // 线型
                 vector<Vec4i>(),  // 轮廓结构信息，索引编号
                 0, Point());
  }
  // cout << "result x " << result.at(1).at(1).x << endl;
  // cout << "result y " << result.at(1).at(1).y << endl;

  // cout << "size2  " << contours.size() << endl;

  imshow("contours", drawing);
  return result;
}

void Differ() {
  string image_name, differ_name, u8_name;
  uint count = 1;
  uint count1 = 1;
  Mat dep_size = imread("1_depth_image.png", IMREAD_ANYDEPTH);
  Mat dep_total = Mat::zeros(dep_size.size(), CV_16UC1);
  Mat dep_average = Mat::zeros(dep_size.size(), CV_16UC1);
  Mat differ;
  // 均值
  for (uint i = 1; i < (IMAGE_NUM + 1); i++) {
    image_name = to_string(i) + "_depth_image.png";
    Mat dep = imread(image_name, IMREAD_ANYDEPTH);
    dep_total += dep;
  }
  dep_average = dep_total / IMAGE_NUM;
  cout << dep_average << endl;
  imwrite("dep_average.png", dep_average);

  // 每一帧与均值的偏差
  for (uint i = 1; i < (IMAGE_NUM + 1); i++) {
    image_name = to_string(i) + "_depth_image.png";
    Mat dep = imread(image_name, IMREAD_ANYDEPTH);
    differ = dep - dep_average;
    differ_name = to_string(i) + "_deffer.png";
    cout << i << endl;
    imwrite(differ_name, differ);
  }

  // 处理偏差
  for (uint j = 1; j < (IMAGE_NUM + 1); j++) {
    image_name = to_string(count) + "_deffer.png";
    Mat differ = imread(image_name, IMREAD_ANYDEPTH);
    //   cout << differ << endl;
    differ.convertTo(differ, CV_8UC1);
    u8_name = to_string(count) + "_deffer_u8.png";
    imwrite(u8_name, differ);
    //   imwrite("defferC8.png", differ1);
    //   cout << differ.at<ushort>(Point(i, j)) << endl;
    //   cout << differ.cols << endl;
    //   cout << differ.rows << endl;
    count++;
  }

  // 处理偏差
  for (uint j = 1; j < (IMAGE_NUM + 1); j++) {
    image_name = to_string(count1) + "_depth_image.png";
    Mat differ = imread(image_name, IMREAD_ANYDEPTH);
    //   cout << differ << endl;
    differ.convertTo(differ, CV_8UC1);
    u8_name = to_string(count1) + "_depth_image_u8.png";
    imwrite(u8_name, differ);
    //   imwrite("defferC8.png", differ1);
    //   cout << differ.at<ushort>(Point(i, j)) << endl;
    //   cout << differ.cols << endl;
    //   cout << differ.rows << endl;
    count1++;
  }
}



int main(int argc, char **argv) {
#if 1
  rs2::pipeline pipe;
  rs2::config cfg;

  // 声明深度着色器，以实现深度数据的可视化
  rs2::colorizer color_map;

  // 声明流的流率
  rs2::rates_printer printer;

  string depth_name;
  uint depth_count = 1;
  Mat depth_u8;
  Mat filter_image_u8;
  Mat depth_average = imread("depth_average.png", IMREAD_ANYDEPTH);
  Mat media;
  medianBlur(depth_average, media, 5);
  Mat depth_temp = Mat::zeros(depth_average.size(), CV_16UC1);
  rs2::decimation_filter decimation_filter;
  rs2::spatial_filter spatial_filter;
  rs2::temporal_filter temporal_filter;

  // cout << depth_average << endl;

  // 配置并启动管道
  cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_ANY, FRAMERATE);
  cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8,
                    FRAMERATE);
  pipe.start(cfg);

  // filter初始化
  // Set Decimation Filter Option
  if (decimation_filter.supports(rs2_option::RS2_OPTION_FILTER_MAGNITUDE)) {
    rs2::option_range option_range = decimation_filter.get_option_range(
        rs2_option::RS2_OPTION_FILTER_MAGNITUDE);
    decimation_filter.set_option(
        rs2_option::RS2_OPTION_FILTER_MAGNITUDE,
        option_range.min);  // 1(min) is not downsampling
  }

  // Set Spatial Filter Option
  if (spatial_filter.supports(rs2_option::RS2_OPTION_HOLES_FILL)) {
    rs2::option_range option_range =
        spatial_filter.get_option_range(rs2_option::RS2_OPTION_HOLES_FILL);
    spatial_filter.set_option(rs2_option::RS2_OPTION_HOLES_FILL,
                              option_range.max);  // 5(max) is fill all holes
  }

  // 标定
  while (true) {
    // 暂停程序，直到帧数据到达
    rs2::frameset frames = pipe.wait_for_frames();

    // 获取彩色图像和深度图像的帧
    rs2::frame color = frames.get_color_frame();
    rs2::depth_frame depth = frames.get_depth_frame();

    rs2::frame filtered_frame = depth;

    // 应用抽取滤波器（下采样）
    filtered_frame = decimation_filter.process( filtered_frame );

    // 从深度帧转换视差帧
    rs2::disparity_transform disparity_transform( true );
    filtered_frame = disparity_transform.process( filtered_frame );

    // 应用空间滤镜（保留边缘的平滑，补孔）
    filtered_frame = spatial_filter.process( filtered_frame );

    // 应用时间过滤器（使用多个先前的帧进行平滑处理）
    filtered_frame = temporal_filter.process( filtered_frame );

    // 从视差帧变换深度帧
    rs2::disparity_transform depth_transform( false );
    filtered_frame = depth_transform.process( filtered_frame );

    // unsigned long long frame_number = frames.get_frame_number();
    depth_name = to_string(depth_count) + "_depth_image_filter.png";

    // rs2::frame 转化为 cv::Mat
    // Mat depth_distance = depth_frame_to_meters(pipe, depth);
    Mat depth_image = frame_to_mat(depth);
    Mat color_image = frame_to_mat(color);
    Mat filter_image = frame_to_mat(filtered_frame);

    cout << "head" << endl;
    // cout << depth_image << endl;
    cout << filter_image << endl;
    depth_image.convertTo(depth_u8, CV_8UC1, 255.0 / 10000.0);
    filter_image.convertTo(filter_image_u8, CV_8UC1, 255.0 / 10000.0);

    imshow("depth_u8", depth_u8);
    imshow("filter_image_u8", filter_image_u8);

    imwrite(depth_name, depth_image);

    cout << depth_count << endl;
    // vector<vector<Point>> result;
    // result = find_obstacle(filter_image, 20, 255, 2500);

    // sleep(1);
    depth_count++;
    waitKey(1);
  }

  // 识别
  while (false) {
    // 暂停程序，直到帧数据到达
    rs2::frameset frames = pipe.wait_for_frames();

    // 获取彩色图像和深度图像的帧
    rs2::frame color = frames.get_color_frame();
    rs2::depth_frame depth = frames.get_depth_frame();

    // unsigned long long frame_number = frames.get_frame_number();

    // rs2::frame 转化为 cv::Mat
    // Mat depth_distance = depth_frame_to_meters(pipe, depth);
    Mat depth_image = Mat(depth_average.rows, depth_average.cols, CV_16SC1);
    depth_image = frame_to_mat(depth);
    cout << "flag" << endl;
    // cout << depth_image << endl;

    Mat color_image = frame_to_mat(color);

    imshow("color_image", color_image);

    vector<vector<Point>> result;

    depth_image = depth_average - depth_image;
    MatExpr abs(depth_image);
    // depth_image
    // cout << depth_image << endl;
    result = find_obstacle(depth_image, 20, 255, 1500);

    // depth_image.convertTo(depth_u8, CV_8UC1, 255.0 / 12000.0);

    // imshow("depth", depth_image);
    // imshow("depth_u8", depth_u8);
    waitKey(1);
  }

#endif

//分析数据
#if 0
  while (true) {
    // Mat image_1 = imread("1_depth_image.png", IMREAD_ANYDEPTH);
    // Mat image_2 = imread("2_depth_image.png", IMREAD_ANYDEPTH);
    // Mat image_3 = imread("depth_average.png", IMREAD_ANYDEPTH);

    // Mat differ = image_1 - image_3;
    // cout << differ << endl;
    // Mat media;
    // medianBlur(differ, media, 5);
    // cout << media << endl;
    Mat depth_u8;
    Mat depth_average = imread("depth_average.png", IMREAD_ANYDEPTH);

    Mat depth_image = imread("1_depth_image.png", IMREAD_ANYDEPTH);

    vector<vector<Point>> result;

    depth_image = depth_average - depth_image;
    MatExpr abs(depth_image);
    cout << "head_row" << endl;
    cout << depth_image << endl;
    result = find_obstacle(depth_image, 20, 255, 2500);

    depth_image.convertTo(depth_u8, CV_8UC1, 255.0 / 5000.0);

    imshow("depth", depth_image);
    imshow("depth_u8", depth_u8);
    waitKey(1);
  }
#endif
  return 1;
}
