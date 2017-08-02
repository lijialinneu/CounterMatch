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


//构造Line
vector<Line> createLine(vector<Vec4i> lines) {
    vector<Line> LineSet;
    for(int i = 0; i < lines.size(); i++) {
        Line *line = new Line(lines[i]);
        LineSet.push_back(*line);
    }
    return LineSet;
}


//计算中点
Point getMidPoint(Line line) {
    double mid_x = (line.start.x + line.end.x) / 2;
    double mid_y = (line.start.y + line.end.y) / 2;
    Point *mid = new Point(mid_x, mid_y);
    return *mid;
}


//计算两个点的距离
double positionDiff(Point p1, Point p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}


//计算平均斜率（用于计算TK）
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


//计算TP
double getTP(InputArray m1, InputArray m2) {
    return (m1.getMat().rows + m2.getMat().rows) / 6;
}


//计算两条直线夹角
double getAngle(double k1, double k2) {
    return atan( abs(k2 - k1) / (1 + k1 * k2) );
}


/*
 计算两个矩阵的相似度
 matlab中的corr2()函数，好麻烦
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

    //计算分子
    double numerator = 0;
    for(int i = 0; i < m1.size(); i++) {
        for(int j = 0; j < m1[i].size(); j++) {
            numerator += (m1[i][j] - mean1) * (m2[i][j] - mean2);
        }
        for(int j = m1[i].size(); j <= m1.size(); j++) {
            numerator += mean1 * mean2;
        }
    }

    //计算分母 sqrt(pow(x,2) + pow(y,2));
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
  产生随机数
*/
int getRandom() {
	return rand() % 255;
}


/*
  利用点到直线距离
  计算两条直线的距离
*/
double distanceBetweenLine(Line l1, Line l2, Mat dst) {
	Point mid = getMidPoint(l1);
	double A = l2.k, B = -1, C = -(l2.k * l2.start.x - l2.start.y);
	double distance =  abs(A * mid.x + B * mid.y + C) / sqrt(A * A + B * B);
	return distance;
}


/*
  判断两条直线是否具备聚合条件
*/
bool canCluster(Line l1, Line l2, int th, Mat dst) {
    return abs(l1.k - l2.k) <= 1 && 
		distanceBetweenLine(l1, l2, dst) < th;
}


/*
  判断两个点是否相近
*/
bool isPointNear(Point p1, Point p2, int th){
	return (abs(p1.x - p2.x) < th && abs(p1.y - p2.y) < th);
}


/*
  判断首尾是否相接，并返回相连的类型
  0：不相连
  1：l1的end 和 l2的start   相连
  2：l1的end 和 l2的end     相连
  3：l1的start 和 l2的start 相连
  4：l1的start 和 l2的end   相连
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
  产生首尾相连的长直线
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
  直线聚合函数
  优先聚合的原则：
  1. 如果两个直线的起点和终点相似，则保留那条长直线
  2. 如果两个直线的首尾能相接，则合并成一条长直线

  由于直线数量不多，采用暴力求解的方法，时间复杂度O(n2)
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

			if(canCluster(line1, line2, th, dst)) { // 判断直线是否具备聚合条件
			    int type = isConnect(line1, line2, th);
			    if(type != 0) {  // 判断是否是连接型
				    Line l = createConnectLine(line1, line2, type);				    
				    (*result).push_back(l);   
					line( dst, l.start, l.end, Scalar(0, 255, 0), 2, CV_AA);				
				    break;
			    }else { // 判断是否是合并型
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
		if(flag || i == length - 1) { // 如果是屋里憋屈型，就把line1 puch进vector
			flag = false;
			line( dst, line1.start, line1.end, Scalar(255, 0, 0), 2, CV_AA);
		    (*result).push_back(line1);
			line_set.insert(i);
		}	
	}
	return *result;
}


/*
  计算两组直线的匹配度
  输入：两个图像的两组直线 lines1，lines2
  算法步骤如下：
  1. 计算每组直线的斜率，计算斜率阈值TK、距离阈值TP
  2. 根据斜率、距离的差值是否满足阈值，找到最佳匹配直线对
  3. 计算每组中的直线与本组中的其他直线之间的夹角
  4. 计算夹角矩阵之间的相似度，并把这个相似度，作为直线的匹配度，返回

  TODO:
  1. 直线匹配时，存在一些短直线相互之间离得很近，并且方向角度相似，其实是一条直线，应当聚类
  2. 通过直线构造三种直线组合，利用直线组合还原高级特折，通过高级特征图匹配

*/
double match(vector<Vec4i> lines1, vector<Vec4i> lines2, InputArray m1, InputArray m2) {
    
    // step1. 对每一条直线计算斜率 
    vector<Line> lineSet1 = createLine(lines1);
    vector<Line> lineSet2 = createLine(lines2);

	// 直线聚合 2017-07-17
	int threshold = 5;
	Mat dst1(m1.getMat().rows,m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	Mat dst2(m1.getMat().rows,m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	lineSet1 = clusterLines(lineSet1, threshold, dst1);
	lineSet2 = clusterLines(lineSet2, threshold, dst2);

	// 画出聚合后的图像
	for(int i = 0; i < lineSet1.size(); i++) {
        Line l = lineSet1[i];
		//cout << "("<<l.start.x << "," << l.start.y << "), (" << l.end.x << "," << l.end.y << ")"<< endl;
        line( dst1, l.start, l.end, Scalar(0, 0, 255), 1, CV_AA);
    }
	imshow("直线聚合后的图像1", dst1);
	for(int i = 0; i < lineSet2.size(); i++) {
        Line l = lineSet2[i];
		//cout << "("<<l.start.x << "," << l.start.y << "), (" << l.end.x << "," << l.end.y << ")"<< endl;
        line( dst2, l.start, l.end, Scalar(0, 0, 255), 1, CV_AA);
    }
    imshow("直线聚合后的图像2", dst2);

	
    //计算平均k
    double t1 = averageK(lineSet1);
    double t2 = averageK(lineSet2);
    double TK = t1 == t2 ? abs(t1) : abs(t1 - t2);
    double TP = getTP(m1, m2);
    if(TK <= 0) return 0.0;

    // step2. 根据斜率、距离之间的差值，配对
    vector<vector<Line>> *pairSet = new vector<vector<Line>>();

    for(int i = 0; i < lineSet1.size(); i++) {
        Line line1 = lineSet1[i];
        double min_diff = MAX;
        int index = 0;
        Line *min_line = new Line();
        
        for(int j = 0; j < lineSet2.size(); j++) {
            Line line2 = lineSet2[j];
            if(abs(line1.k - line2.k) < TK) { //判断1. 斜率差值在TK内
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
        
        if(min_diff < TP) { //判断2. 距离差值在TP内
            vector<Line> *v = new vector<Line>();
            (*v).push_back(line1);
            (*v).push_back(*min_line);
            (*pairSet).push_back(*v);
        }
        delete min_line;

    }

    //画出图像，便于分析
    Mat src1(m1.getMat().rows,m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
    Mat src2(m2.getMat().rows,m2.getMat().cols, CV_8UC3, Scalar(255,255,255));

    for(int i = 0; i < (*pairSet).size(); i++) {
		//产生三个随机数
		int b = getRandom();
		int g = getRandom();
		int r = getRandom();
		
        vector<Line> v = (*pairSet)[i];
        line( src1, v[0].start, v[0].end, Scalar(b, g, r), 3, CV_AA);
        line( src2, v[1].start, v[1].end, Scalar(b, g, r), 3, CV_AA);
    }
    imshow("1", src1);
    imshow("2", src2);

    //计算直线之间的误差O(n2)
    //计算直线与本张图像中的其他直线的夹角
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

    // 计算夹角矩阵的相似度
    double rate = calculateCorr2((*angles_list1),(*angles_list2));
    rate *= (double)(*pairSet).size() / lineSet1.size();


	// TODO外立面的构建



    return rate;

}