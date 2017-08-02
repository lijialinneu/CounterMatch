#include "Line.h"
#include "MinHeap.h"
#include<opencv2/opencv.hpp>
#include<math.h>

#define PI 3.1415926
#define MAX 100000

using namespace cv;

Line::Line(Vec4i l) {
    line = l;
    
    // �趨�˵�
    start = Point(l[0], l[1]);
    end = Point(l[2], l[3]);

    // ���㳤�Ⱥ���Գ���
    length = MinHeap::getLength(line);
    len = length / height;
    
    // ����ֱ�ߵ�б��
    if(end.x - start.x != 0) {
        k = (double)(start.y - end.y) / (start.x - end.x);
    }else {
        k = MAX;
    }

    //����ֱ����ˮƽ����ļн�
    theta = atan(k) * 180 / PI;
}

Line::Line(void) {
}

Line::~Line(void){ 
}
