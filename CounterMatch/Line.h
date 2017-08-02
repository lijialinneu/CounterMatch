#pragma once
#include<opencv2/opencv.hpp>

using namespace cv;

class Line
{
public:
    double width;  //����ͼ��Ŀ�
    double height; //����ͼ��ĸ�
    Vec4i line;  //ֱ�߱���
    Point start; //ֱ�ߵ����
    Point end;   //ֱ�ߵ��յ�
    double length; //ֱ�ߵĳ���
    double len; //ֱ�ߵ���Գ��ȣ����ͼ��ĸߣ�
    double k; //ֱ�ߵ�б��
    double theta; //ֱ����ˮƽ����ļн�
    vector<double> angles; //�н�����
    vector<Line> parallels; //ƽ���߼���
    vector<Line> verticals; //��ֱ�߼���

    Line(Vec4i line);
    Line(void);
    ~Line(void);
};