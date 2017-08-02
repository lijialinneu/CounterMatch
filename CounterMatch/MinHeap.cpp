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
    //delete (*minheap); //释放空间
}

// 创建小顶堆
void MinHeap::createMinHeap(vector<Vec4i> a) {
    for(int i = 0; i < maxsize; i++) {
        minheap.push_back(a[i]);
    }
}

//插入元素
void MinHeap::insert(Vec4i line) {
    if(getLength(line) > getLength(getTop())) {
        minheap[0] = line;
        filterDown(0);
    }
}

//向下调整
void MinHeap::filterDown(int current) {
    int end = minheap.size()-1;
    int child = current * 2 + 1; //当前节点的左孩子
    Vec4i line = minheap[current]; //保存当前节点

    while(child <= end) {
        // 选出两个孩子中的较小孩子
        if(child < end && getLength(minheap[child+1]) < getLength(minheap[child]))
            child++;
        if(getLength(line) < getLength(minheap[child])) break;
        else {
            minheap[current] = minheap[child];//孩子节点覆盖当前节点
            current = child;
            child = child * 2 + 1;
        }
    }
    minheap[current] = line;
}

//获取堆顶元素
Vec4i MinHeap::getTop() {
    if(minheap.size() != 0)
        return minheap[0];
    return NULL;
}

//计算距离
double MinHeap::getLength(Vec4i line) {
    double x = line[0]-line[2];
    double y = line[1]-line[3];
    return sqrt(pow(x,2) + pow(y,2));
}

// 获取堆中的全部元素
vector<Vec4i> MinHeap::getHeap() {
    vector<Vec4i> heap;
    for(int i = 0; i < minheap.size(); i++)
        heap.push_back(minheap[i]);
    return heap;
}


//打印小顶堆
//void MinHeap::print() {
//    for(int i = 0; i < minheap.size(); i++)
//        cout << minheap[i] << " ";
//    cout << endl;
//}
//向上调整堆
//void MinHeap::filterUp(int index) {
//    Vec4i line = minheap[index]; //记录下当前节点
//    while(index > 0) {
//        int indexparent = (index - 1) / 2; //得到其双亲
//        if(getLength(line) < getLength(minheap[indexparent])) {
//            break;
//        }else { //交换两节点
//            //新节点被其父节点覆盖
//            minheap[index] = minheap[indexparent];
//            index = indexparent;
//        }
//    }
//    minheap[index] = line; //给新节点赋值
//}


////删除元素
//void MinHeap::remove(Vec4i key) {
//    if(minheap.size() == 0) return ;
//    int index; //被删除的元素下标
//    for(index = 0; index < minheap.size(); index++) {
//        if(minheap[index] == key) break; //找到被删除元素的下标
//    }
//    
//    if(index == minheap.size()) return ; //数组中没有要删除的元素
//
//    //使用数组的最后一个元素替代被删除的元素
//    minheap[index] = minheap.back();
//    filterDown(index);
//}