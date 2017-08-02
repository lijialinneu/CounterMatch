#include"MinHeap.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
//#define getLength(line) sqrt(pow((line[0])-(line[2]),2) + \
//                           pow((line[1])-(line[3]),2));

using namespace std;
using namespace cv;

MinHeap::MinHeap(int k) {
    maxsize = k;
}

MinHeap::~MinHeap() {
    minheap.clear();
    //delete (*minheap); //�ͷſռ�
}

// ����С����
void MinHeap::createMinHeap(vector<Vec4i> a) {
    for(int i = 0; i < maxsize; i++) {
        minheap.push_back(a[i]);
    }
}

//����Ԫ��
void MinHeap::insert(Vec4i line) {
    if(getLength(line) > getLength(getTop())) {
        minheap[0] = line;
        filterDown(0);
    }
}

//���µ���
void MinHeap::filterDown(int current) {
    int end = minheap.size()-1;
    int child = current * 2 + 1; //��ǰ�ڵ������
    Vec4i line = minheap[current]; //���浱ǰ�ڵ�

    while(child <= end) {
        // ѡ�����������еĽ�С����
        if(child < end && getLength(minheap[child+1]) < getLength(minheap[child]))
            child++;
        if(getLength(line) < getLength(minheap[child])) break;
        else {
            minheap[current] = minheap[child];//���ӽڵ㸲�ǵ�ǰ�ڵ�
            current = child;
            child = child * 2 + 1;
        }
    }
    minheap[current] = line;
}

//��ȡ�Ѷ�Ԫ��
Vec4i MinHeap::getTop() {
    if(minheap.size() != 0)
        return minheap[0];
    return NULL;
}

//�������
double MinHeap::getLength(Vec4i line) {
    double x = line[0]-line[2];
    double y = line[1]-line[3];
    return sqrt(pow(x,2) + pow(y,2));
}

// ��ȡ���е�ȫ��Ԫ��
vector<Vec4i> MinHeap::getHeap() {
    vector<Vec4i> heap;
    for(int i = 0; i < minheap.size(); i++)
        heap.push_back(minheap[i]);
    return heap;
}


//��ӡС����
//void MinHeap::print() {
//    for(int i = 0; i < minheap.size(); i++)
//        cout << minheap[i] << " ";
//    cout << endl;
//}
//���ϵ�����
//void MinHeap::filterUp(int index) {
//    Vec4i line = minheap[index]; //��¼�µ�ǰ�ڵ�
//    while(index > 0) {
//        int indexparent = (index - 1) / 2; //�õ���˫��
//        if(getLength(line) < getLength(minheap[indexparent])) {
//            break;
//        }else { //�������ڵ�
//            //�½ڵ㱻�丸�ڵ㸲��
//            minheap[index] = minheap[indexparent];
//            index = indexparent;
//        }
//    }
//    minheap[index] = line; //���½ڵ㸳ֵ
//}


////ɾ��Ԫ��
//void MinHeap::remove(Vec4i key) {
//    if(minheap.size() == 0) return ;
//    int index; //��ɾ����Ԫ���±�
//    for(index = 0; index < minheap.size(); index++) {
//        if(minheap[index] == key) break; //�ҵ���ɾ��Ԫ�ص��±�
//    }
//    
//    if(index == minheap.size()) return ; //������û��Ҫɾ����Ԫ��
//
//    //ʹ����������һ��Ԫ�������ɾ����Ԫ��
//    minheap[index] = minheap.back();
//    filterDown(index);
//}