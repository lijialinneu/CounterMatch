#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <string.h>
#include <iostream>
#include "countermatch.h"
#include "Line.h"
#include <math.h>
#include <set>

#define MAX 1.7976931348623158e+308
#define MAXK 100000

using namespace cv;
using namespace std;


//����Line
vector<Line> createLine(vector<Vec4i> lines) {
    vector<Line> LineSet;
    for(int i = 0; i < lines.size(); i++) {
        Line *line = new Line(lines[i]);
        LineSet.push_back(*line);
    }
    return LineSet;
}


//�����е�
Point getMidPoint(Line line) {
    double mid_x = (line.start.x + line.end.x) / 2;
    double mid_y = (line.start.y + line.end.y) / 2;
    Point *mid = new Point(mid_x, mid_y);
    return *mid;
}


//����������ľ���
double positionDiff(Point p1, Point p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}


//����ƽ��б�ʣ����ڼ���TK��
double averageK(vector<Line> LineSet) {
    double avg = 0;
    int count = 0;
    for(int i = 0; i < LineSet.size(); i++) {
        if(LineSet[i].k != MAXK) {
            avg += LineSet[i].k;
            count++;
        }
    }
    avg /= count; 
    return avg;
}


//����TP
double getTP(InputArray m1, InputArray m2) {
    return (m1.getMat().rows + m2.getMat().rows) / 6;
}


//��������ֱ�߼н�
double getAngle(double k1, double k2) {
    return atan( abs(k2 - k1) / (1 + k1 * k2) );
}


/*
 ����������������ƶ�
 matlab�е�corr2()���������鷳
*/
double calculateMean(vector<vector<double>> m) {

    vector<double> *mean = new vector<double>();
    int p = 0;
    for(int j = m[p].size()-1; j>=0; j--) {
        double count = 0;
        for(int i = 0, k = j; i <= m[p].size()-1; k--, i++) {
            count += m[i][k];
        }
        count /= m.size();
        (*mean).push_back(count);
        p++;
    }

    double count = 0;
    for(int i = 0; i < (*mean).size(); i++) {
        count += (*mean)[i];
    }
    count /= ((*mean).size()+1);
    return count;
}

double calculateCorr2(vector<vector<double>> m1, 
                      vector<vector<double>> m2) {

    double mean1 = calculateMean(m1);
    double mean2 = calculateMean(m2);

    //�������
    double numerator = 0;
    for(int i = 0; i < m1.size(); i++) {
        for(int j = 0; j < m1[i].size(); j++) {
            numerator += (m1[i][j] - mean1) * (m2[i][j] - mean2);
        }
        for(int j = m1[i].size(); j <= m1.size(); j++) {
            numerator += mean1 * mean2;
        }
    }

    //�����ĸ sqrt(pow(x,2) + pow(y,2));
    double d1 = 0;
    double d2 = 0;
    for(int i = 0; i < m1.size(); i++) {
        for(int j = 0; j < m1[i].size(); j++) {
            d1 += pow((m1[i][j] - mean1),2);
            d2 += pow((m2[i][j] - mean2),2);
        }
        for(int j = m1[i].size(); j <= m1.size(); j++) {
            d1 += pow(mean1,2);
            d2 += pow(mean2,2);
        }
    }
    double denominator = sqrt(d1) * sqrt(d2);

    if(numerator == 0) return 0.0;
    return numerator/denominator;
}

/*
  ���������
*/
int getRandom() {
	return rand() % 255;
}


/*
  ���õ㵽ֱ�߾���
  ��������ֱ�ߵľ���
*/
double distanceBetweenLine(Line l1, Line l2, Mat dst) {
	Point mid = getMidPoint(l1);
	double A = l2.k, B = -1, C = -(l2.k * l2.start.x - l2.start.y);
	double distance =  abs(A * mid.x + B * mid.y + C) / sqrt(A * A + B * B);
	return distance;
}


/*
  �ж�����ֱ���Ƿ�߱��ۺ�����
*/
bool canCluster(Line l1, Line l2, int th, Mat dst) {
    return abs(l1.k - l2.k) <= 1 && 
		distanceBetweenLine(l1, l2, dst) < th;
}


/*
  �ж��������Ƿ����
*/
bool isPointNear(Point p1, Point p2, int th){
	return (abs(p1.x - p2.x) < th && abs(p1.y - p2.y) < th);
}


/*
  �ж���β�Ƿ���ӣ�����������������
  0��������
  1��l1��end �� l2��start   ����
  2��l1��end �� l2��end     ����
  3��l1��start �� l2��start ����
  4��l1��start �� l2��end   ����
*/
int isConnect(Line l1, Line l2, int th) {
	int len = max(l1.length, l2.length);
	if(isPointNear(l1.end, l2.start, th) && !isPointNear(l1.start, l2.end, len)) {
		return 1;
	}else if(isPointNear(l1.end, l2.end, th) && !isPointNear(l1.start, l2.start, len)) {
		return 2;
	}else if(isPointNear(l1.start, l2.start, th) && !isPointNear(l1.end, l2.end, len)) {
		return 3;
	}else if(isPointNear(l1.start, l2.end, th) && !isPointNear(l1.end, l2.start, len)) {
		return 4;
	}
	return 0;
}


/*
  ������β�����ĳ�ֱ��
*/
Line createConnectLine(Line l1, Line l2, int type) {
	Line l;
	switch(type) {
	case 1:
		l.start = l1.start;
		l.end = l2.end;
		break;
    case 2:
		l.start = l1.start;
		l.end = l2.start;
		break;
    case 3:
		l.start = l1.end;
		l.end = l2.end;
		break;
    case 4:
		l.start = l1.end;
		l.end = l2.start;
		break;
	default:
        break;
	}
	return l;
}

/*
  ֱ�߾ۺϺ���
  ���Ⱦۺϵ�ԭ��
  1. �������ֱ�ߵ������յ����ƣ�����������ֱ��
  2. �������ֱ�ߵ���β����ӣ���ϲ���һ����ֱ��

  ����ֱ���������࣬���ñ������ķ�����ʱ�临�Ӷ�O(n2)
*/
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	int length = lines.size();
	set<int> line_set;
	set<int>::iterator iter;

	for(int i = 0; i < length; i++) {

		iter = line_set.find(i);
		if(iter != line_set.end()) {
		    continue;
		}
		Line line1 = lines[i];
		bool flag = false;
		
		for(int j = i + 1; j < length; j++) {			
			iter = line_set.find(j);
		    if(iter != line_set.end()) {
		        continue;
		    }
		    Line line2 = lines[j];

			if(canCluster(line1, line2, th, dst)) { // �ж�ֱ���Ƿ�߱��ۺ�����
			    int type = isConnect(line1, line2, th);
			    if(type != 0) {  // �ж��Ƿ���������
				    Line l = createConnectLine(line1, line2, type);				    
				    (*result).push_back(l);   
					line( dst, l.start, l.end, Scalar(0, 255, 0), 2, CV_AA);				
				    break;
			    }else { // �ж��Ƿ��Ǻϲ���
					if(line1.length > line2.length) {
					    (*result).push_back(line1);
					}else {
					    (*result).push_back(line2);
					}
				    break;
			    } 
				line_set.insert(i);
				line_set.insert(j);
			}
			if(j == length - 1) {
			    flag = true;
			}
		}
		if(flag || i == length - 1) { // �������������ͣ��Ͱ�line1 puch��vector
			flag = false;
			line( dst, line1.start, line1.end, Scalar(255, 0, 0), 2, CV_AA);
		    (*result).push_back(line1);
			line_set.insert(i);
		}	
	}
	return *result;
}


/*
  ��������ֱ�ߵ�ƥ���
  ���룺����ͼ�������ֱ�� lines1��lines2
  �㷨�������£�
  1. ����ÿ��ֱ�ߵ�б�ʣ�����б����ֵTK��������ֵTP
  2. ����б�ʡ�����Ĳ�ֵ�Ƿ�������ֵ���ҵ����ƥ��ֱ�߶�
  3. ����ÿ���е�ֱ���뱾���е�����ֱ��֮��ļн�
  4. ����нǾ���֮������ƶȣ�����������ƶȣ���Ϊֱ�ߵ�ƥ��ȣ�����

  TODO:
  1. ֱ��ƥ��ʱ������һЩ��ֱ���໥֮����úܽ������ҷ���Ƕ����ƣ���ʵ��һ��ֱ�ߣ�Ӧ������
  2. ͨ��ֱ�߹�������ֱ����ϣ�����ֱ����ϻ�ԭ�߼����ۣ�ͨ���߼�����ͼƥ��

*/
double match(vector<Vec4i> lines1, vector<Vec4i> lines2, InputArray m1, InputArray m2) {
    
    // step1. ��ÿһ��ֱ�߼���б�� 
    vector<Line> lineSet1 = createLine(lines1);
    vector<Line> lineSet2 = createLine(lines2);

	// ֱ�߾ۺ� 2017-07-17
	int threshold = 5;
	Mat dst1(m1.getMat().rows,m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	Mat dst2(m1.getMat().rows,m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	lineSet1 = clusterLines(lineSet1, threshold, dst1);
	lineSet2 = clusterLines(lineSet2, threshold, dst2);

	// �����ۺϺ��ͼ��
	for(int i = 0; i < lineSet1.size(); i++) {
        Line l = lineSet1[i];
		//cout << "("<<l.start.x << "," << l.start.y << "), (" << l.end.x << "," << l.end.y << ")"<< endl;
        line( dst1, l.start, l.end, Scalar(0, 0, 255), 1, CV_AA);
    }
	imshow("ֱ�߾ۺϺ��ͼ��1", dst1);
	for(int i = 0; i < lineSet2.size(); i++) {
        Line l = lineSet2[i];
		//cout << "("<<l.start.x << "," << l.start.y << "), (" << l.end.x << "," << l.end.y << ")"<< endl;
        line( dst2, l.start, l.end, Scalar(0, 0, 255), 1, CV_AA);
    }
    imshow("ֱ�߾ۺϺ��ͼ��2", dst2);

	
    //����ƽ��k
    double t1 = averageK(lineSet1);
    double t2 = averageK(lineSet2);
    double TK = t1 == t2 ? abs(t1) : abs(t1 - t2);
    double TP = getTP(m1, m2);
    if(TK <= 0) return 0.0;

    // step2. ����б�ʡ�����֮��Ĳ�ֵ�����
    vector<vector<Line>> *pairSet = new vector<vector<Line>>();

    for(int i = 0; i < lineSet1.size(); i++) {
        Line line1 = lineSet1[i];
        double min_diff = MAX;
        int index = 0;
        Line *min_line = new Line();
        
        for(int j = 0; j < lineSet2.size(); j++) {
            Line line2 = lineSet2[j];
            if(abs(line1.k - line2.k) < TK) { //�ж�1. б�ʲ�ֵ��TK��
                Point mid1 = getMidPoint(line1);
                Point mid2 = getMidPoint(line2);
                double diff = positionDiff(mid1, mid2);
                if(diff < min_diff) {
                    min_diff = diff;
                    index = j;
                    *min_line = line2;
                }
            }
        }
        
        if(min_diff < TP) { //�ж�2. �����ֵ��TP��
            vector<Line> *v = new vector<Line>();
            (*v).push_back(line1);
            (*v).push_back(*min_line);
            (*pairSet).push_back(*v);
        }
        delete min_line;

    }

    //����ͼ�񣬱��ڷ���
    Mat src1(m1.getMat().rows,m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
    Mat src2(m2.getMat().rows,m2.getMat().cols, CV_8UC3, Scalar(255,255,255));

    for(int i = 0; i < (*pairSet).size(); i++) {
		//�������������
		int b = getRandom();
		int g = getRandom();
		int r = getRandom();
		
        vector<Line> v = (*pairSet)[i];
        line( src1, v[0].start, v[0].end, Scalar(b, g, r), 3, CV_AA);
        line( src2, v[1].start, v[1].end, Scalar(b, g, r), 3, CV_AA);
    }
    imshow("1", src1);
    imshow("2", src2);

    //����ֱ��֮������O(n2)
    //����ֱ���뱾��ͼ���е�����ֱ�ߵļн�
    vector<vector<double>> *angles_list1 = new vector<vector<double>>(); 
    vector<vector<double>> *angles_list2 = new vector<vector<double>>();

    for(int i = 0; i < (*pairSet).size(); i++) {
        vector<Line> v1 = (*pairSet)[i];
        vector<double> *angles1 = new vector<double>();
        vector<double> *angles2 = new vector<double>();

        for(int j = i+1; j < (*pairSet).size(); j++) {
            vector<Line> v2 = (*pairSet)[j];
            (*angles1).push_back(getAngle(v1[0].k, v2[0].k));
            (*angles2).push_back(getAngle(v1[1].k, v2[1].k));
        }
        (*angles_list1).push_back(*angles1);
        (*angles_list2).push_back(*angles2);
        delete angles1;
        delete angles2;
    }

    // ����нǾ�������ƶ�
    double rate = calculateCorr2((*angles_list1),(*angles_list2));
    rate *= (double)(*pairSet).size() / lineSet1.size();


	// TODO������Ĺ���



    return rate;

}