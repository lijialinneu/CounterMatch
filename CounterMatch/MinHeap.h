#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

/*С����, ���ڵ��ֵС�ں��ӵ�ֵ
������Ҫ������һ�Ѻ����������ҳ�����ǰk�����ݣ�
��õķ���������С������ֱ����ǰk�����ݽ���һ��С���ѣ�Ȼ�����ʣ�������
��������� < �Ѷ�Ԫ��, ˵������k��������С������ҪС����ֱ������������������һ������
��������� > �Ѷ��������򽫴����ͶѶ�����������Ȼ��ӶѶ����µ����ѣ�ʹ����������С���ѡ�
*/

class MinHeap 
{
private:
    int maxsize; //�ѵĴ�С
    //void filterUp(int index); //���ϵ�����
    void filterDown(int begin); //���ϵ�����
    vector<Vec4i> minheap;

public:
    MinHeap(int k);
    ~MinHeap();

    void insert(Vec4i val); //����Ԫ��
    //void remove(Vec4i val); //ɾ��Ԫ��
    //void print(); // ��ӡ�����
    Vec4i getTop(); //��ȡ�Ѷ�Ԫ��
    void createMinHeap(vector<Vec4i> a);
    static double getLength(Vec4i line);//���㳤��
    vector<Vec4i> getHeap(); //��ȡ���е�ȫ��Ԫ��
};