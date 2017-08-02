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
 * ȡ��topK��ֱ��
 */
vector<Vec4i> getTopK(vector<Vec4i> lines, int K) {
    
    // MinHeap heap(K);
    MinHeap* heap = new MinHeap(K);

	// �����󶥶�
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
    //Ϊ����ֱ��ͼ���ñ���  
    //��������Ҫ�����ͼ���ͨ����������Ҫ����ͼ����ĸ�ͨ����bgr�ռ���Ҫȷ������ b��g��r�ռ䣩  
    int channels = 0;  
    //Ȼ������������Ľ���洢�� �ռ� ����MatND�������洢���  
    MatND dstHist;  
    //��������ֱ��ͼ��ÿһ��ά�ȵ� ��������Ŀ�����ǽ���ֵ���飬���ж����飩  
    int histSize[] = { 256 };       //�������д��int histSize = 256;   ��ô������ü���ֱ��ͼ�ĺ�����ʱ�򣬸ñ�����Ҫд &histSize  
    //�����ȷ��ÿ��ά�ȵ�ȡֵ��Χ�����Ǻ����������  
    //���ȵö���һ�����������洢 ����ά�ȵ� ��ֵ��ȡֵ��Χ  
    float midRanges[] = { 0, 256 };  
    const float *ranges[] = { midRanges };  
  
    calcHist(&srcImage, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);  
  
    //calcHist  �������ý�����dstHist�����н������� ֱ��ͼ����Ϣ  ��dstHist��ģ�溯�� at<Type>(i)�õ���i��������ֵ  
    //at<Type>(i, j)�õ���i�����ҵ�j��������ֵ  
  
    //��ʼֱ�۵���ʾֱ��ͼ��������ֱ��ͼ  
    //�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��  
    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);  
    //��Ϊ�κ�һ��ͼ���ĳ�����ص��ܸ��������п��ܻ��кܶ࣬�ᳬ���������ͼ��ĳߴ磬�������������ȶԸ������з�Χ������  
    //���� minMaxLoc�������õ�����ֱ��ͼ������ص�������  
    double g_dHistMaxValue;  
    minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);  
    //�����صĸ������ϵ� ͼ������Χ��  
    //����ֱ��ͼ�õ�������  
    for (int i = 0; i < 256; i++)  
    {  
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);  
  
        line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));  
    }  
  
    imshow("��ֱ��ͼ��", drawImage);  
}


int main(){
	
    clock_t start, finish;
    double totaltime;
    start = clock(); // �����ʱ

    string path1 = "images/test3.jpg";
    string path2 = "images/test4.jpg";


    // step1. ͼƬԤ����
    Mat src1 = imread(path1);
    Mat src2 = imread(path2);
    imshow("��ԭʼͼ1��", src1);
    imshow("��ԭʼͼ2��", src2);


	//-----------------------------------------
	//  ��1�� ֱ��ͼ����ͼ��ͼ��ͬѡ�ò�ͬ������
	//-----------------------------------------
	//drawHist(src1);

    // ͼ���С�仯
    //double height = (double)src1.rows / src1.cols * src2.cols;
    //resize(src1, src1, Size(src2.cols, height), 0, 0, CV_INTER_LINEAR);

    Mat edge1, edge2; // ����Ŀ�����

    blur(src1, edge1, Size(3,3) ); // ʹ�� 3x3�ں�������
    blur(src2, edge2, Size(3,3) );
    
	// step2. ��Ե���
    //myCanny(edge1, edge1, 100, 300, 3);// ����Canny����
    //myCanny(edge2, edge2, 50, 150, 3);
	Canny(edge1, edge1, 50, 150, 3);
	Canny(edge2, edge2, 50, 150, 3);

    imshow("����Եͼ1��", edge1);
	imshow("����Եͼ2��", edge2);

    // step3. �ֱ������ͼƬ����ֱ�߼��
    Mat dst1, dst2;
    dst1.create(src1.size(), src1.type());
    dst2.create(edge2.size(), edge2.type());
    vector<Vec4i> lines1, lines2;

    myHoughLinesP(edge1, lines1, 1, CV_PI/180, 50, 50, 10 );
    myHoughLinesP(edge2, lines2, 1, CV_PI/180, 50, 50, 10 );

    // get TopK ��ֱ��
    const int K = 50;
    lines1 = getTopK(lines1, K);
    lines2 = getTopK(lines2, K);

	// ����ֱ��
    for(int i = 0; i < lines1.size(); i++ ) {
        Vec4i l = lines1[i];
        line( dst1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }

    for(int i = 0; i < lines2.size(); i++ ) {
        Vec4i l = lines2[i];
        line( dst2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }

    imshow("��Houghֱ����ȡ1��", dst1);
	imshow("��Houghֱ����ȡ2��", dst2);

    //TODO ����ֱ�߼��ƥ������
    double rate = match(lines1, lines2, dst1, dst2);
	cout<<"ƥ����ǣ�"<<rate<<endl;

    finish = clock();
    totaltime = (double) (finish - start)/CLOCKS_PER_SEC;
    cout<<"��ʱ" <<totaltime <<"��"<<endl;

	
	waitKey(0); 
    return 0; 
}
