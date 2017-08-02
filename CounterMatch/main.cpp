#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<time.h>
#include<iostream>
#include<math.h>
#include"countermatch.h"
#include"MinHeap.h"
#include"countermatch.h"

using namespace cv;
using namespace std;

/*
 * 取得topK的直线
 */
vector<Vec4i> getTopK(vector<Vec4i> lines, int K) {
    
    // MinHeap heap(K);
    MinHeap* heap = new MinHeap(K);

	// 创建大顶堆
    (*heap).createMinHeap(lines);
    for(int i = K+1; i< lines.size(); i++) {
        (*heap).insert(lines[i]);
    }
    // (*heap).print();
    lines = (*heap).getHeap();
    delete heap;
    return lines;
}

void drawHist(Mat srcImage) {
    //为计算直方图配置变量  
    //首先是需要计算的图像的通道，就是需要计算图像的哪个通道（bgr空间需要确定计算 b或g货r空间）  
    int channels = 0;  
    //然后是配置输出的结果存储的 空间 ，用MatND类型来存储结果  
    MatND dstHist;  
    //接下来是直方图的每一个维度的 柱条的数目（就是将数值分组，共有多少组）  
    int histSize[] = { 256 };       //如果这里写成int histSize = 256;   那么下面调用计算直方图的函数的时候，该变量需要写 &histSize  
    //最后是确定每个维度的取值范围，就是横坐标的总数  
    //首先得定义一个变量用来存储 单个维度的 数值的取值范围  
    float midRanges[] = { 0, 256 };  
    const float *ranges[] = { midRanges };  
  
    calcHist(&srcImage, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);  
  
    //calcHist  函数调用结束后，dstHist变量中将储存了 直方图的信息  用dstHist的模版函数 at<Type>(i)得到第i个柱条的值  
    //at<Type>(i, j)得到第i个并且第j个柱条的值  
  
    //开始直观的显示直方图——绘制直方图  
    //首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像  
    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);  
    //因为任何一个图像的某个像素的总个数，都有可能会有很多，会超出所定义的图像的尺寸，针对这种情况，先对个数进行范围的限制  
    //先用 minMaxLoc函数来得到计算直方图后的像素的最大个数  
    double g_dHistMaxValue;  
    minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);  
    //将像素的个数整合到 图像的最大范围内  
    //遍历直方图得到的数据  
    for (int i = 0; i < 256; i++)  
    {  
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);  
  
        line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));  
    }  
  
    imshow("【直方图】", drawImage);  
}


int main(){
	
    clock_t start, finish;
    double totaltime;
    start = clock(); // 程序计时

    string path1 = "images/test3.jpg";
    string path2 = "images/test4.jpg";


    // step1. 图片预处理
    Mat src1 = imread(path1);
    Mat src2 = imread(path2);
    imshow("【原始图1】", src1);
    imshow("【原始图2】", src2);


	//-----------------------------------------
	//  【1】 直方图分析图像，图像不同选用不同的算子
	//-----------------------------------------
	//drawHist(src1);

    // 图像大小变化
    //double height = (double)src1.rows / src1.cols * src2.cols;
    //resize(src1, src1, Size(src2.cols, height), 0, 0, CV_INTER_LINEAR);

    Mat edge1, edge2; // 创建目标矩阵

    blur(src1, edge1, Size(3,3) ); // 使用 3x3内核来降噪
    blur(src2, edge2, Size(3,3) );
    
	// step2. 边缘检测
    //myCanny(edge1, edge1, 100, 300, 3);// 运行Canny算子
    //myCanny(edge2, edge2, 50, 150, 3);
	Canny(edge1, edge1, 50, 150, 3);
	Canny(edge2, edge2, 50, 150, 3);

    imshow("【边缘图1】", edge1);
	imshow("【边缘图2】", edge2);

    // step3. 分别对两张图片进行直线检测
    Mat dst1, dst2;
    dst1.create(src1.size(), src1.type());
    dst2.create(edge2.size(), edge2.type());
    vector<Vec4i> lines1, lines2;

    myHoughLinesP(edge1, lines1, 1, CV_PI/180, 50, 50, 10 );
    myHoughLinesP(edge2, lines2, 1, CV_PI/180, 50, 50, 10 );

    // get TopK 条直线
    const int K = 50;
    lines1 = getTopK(lines1, K);
    lines2 = getTopK(lines2, K);

	// 绘制直线
    for(int i = 0; i < lines1.size(); i++ ) {
        Vec4i l = lines1[i];
        line( dst1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }

    for(int i = 0; i < lines2.size(); i++ ) {
        Vec4i l = lines2[i];
        line( dst2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }

    imshow("【Hough直线提取1】", dst1);
	imshow("【Hough直线提取2】", dst2);

    //TODO 根据直线检测匹配轮廓
    double rate = match(lines1, lines2, dst1, dst2);
	cout<<"匹配度是："<<rate<<endl;

    finish = clock();
    totaltime = (double) (finish - start)/CLOCKS_PER_SEC;
    cout<<"用时" <<totaltime <<"秒"<<endl;

	
	waitKey(0); 
    return 0; 
}
