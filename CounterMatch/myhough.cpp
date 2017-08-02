#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define halfPi ((float)(CV_PI*0.5))
#define Pi     ((float)CV_PI)
#define a0  0 /*-4.172325e-7f*/   /*(-(float)0x7)/((float)0x1000000); */
#define a1 1.000025f        /*((float)0x1922253)/((float)0x1000000)*2/Pi; */
#define a2 -2.652905e-4f    /*(-(float)0x2ae6)/((float)0x1000000)*4/(Pi*Pi); */
#define a3 -0.165624f       /*(-(float)0xa45511)/((float)0x1000000)*8/(Pi*Pi*Pi); */
#define a4 -1.964532e-3f    /*(-(float)0x30fd3)/((float)0x1000000)*16/(Pi*Pi*Pi*Pi); */
#define a5 1.02575e-2f      /*((float)0x191cac)/((float)0x1000000)*32/(Pi*Pi*Pi*Pi*Pi); */
#define a6 -9.580378e-4f    /*(-(float)0x3af27)/((float)0x1000000)*64/(Pi*Pi*Pi*Pi*Pi*Pi); */

#define _sin(x) ((((((a6*(x) + a5)*(x) + a4)*(x) + a3)*(x) + a2)*(x) + a1)*(x) + a0)
#define _cos(x) _sin(halfPi - (x))

/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

static void
icvHoughLinesProbabilistic( CvMat* image,
                            float rho, float theta, int threshold,
                            int lineLength, int lineGap,
                            CvSeq *lines, int linesMax ) {

    //rho:��λ���ؾ��ȣ���ȡ1�����irho��Ϊ1
    //theta ��λ����
    //�ۼ�ƽ����Կ�����rho���غ�theta������ɵĶ�άֱ��ͼ
    //linesMax��ʾ֧�������ص�ֱ�ߵĵ������

    Mat accum, mask;       // accum ���������� maskΪԴ�����
    vector<float> trigtab; //���ڴ洢���ȼ���õ����Һ�����
    MemStorage storage(cvCreateMemStorage(0)); //����һ���ڴ�ռ�

    CvSeq* seq;            //���ڴ洢��Ե����
    CvSeqWriter writer;
    int width, height;
    int numangle, numrho;  //rho����ɢ������theta����ɢ������theta�������ֵ�� 
    float ang;
    int r, n, count;
    CvPoint pt;
    float irho = 1 / rho;  //rho:��λ���ؾ��ȣ���ȡ1�����irho��Ϊ1
    CvRNG rng = cvRNG(-1); //�����
    const float* ttab;     //����trigtab�ĵ�ַָ��
    uchar* mdata0;         //����mask�ĵ�ַָ��
    
    CV_Assert( CV_IS_MAT(image) && CV_MAT_TYPE(image->type) == CV_8UC1 );

    width = image->cols;   //��ȡ����ͼ�����������ȣ�
    height = image->rows;  //��ȡ����ͼ����������߶ȣ�

    //���ݾ��ȼ�����Ƕȵ����ֵ
    numangle = cvRound(CV_PI / theta); 
    //���ݾ��ȼ����r�����ֵ
    numrho = cvRound(((width + height) * 2 + 1) / rho); //Ϊʲô��ô��?

    accum.create( numangle, numrho, CV_32SC1 ); // �����ۼ������󣬼�Hough�ռ�
    mask.create( height, width, CV_8UC1 );      //�����������

    //����trigtab�Ĵ�С����ΪҪ�洢sin��cos�����Գ���Ϊ�Ƕ���ɢ����2��
    trigtab.resize(numangle*2); 
    accum = cv::Scalar(0); //�ۼ�����������

    //Ϊ�����ظ����㣬���ȼ����������������Һ�����ֵ
    for( ang = 0, n = 0; n < numangle; ang += theta, n++ ) {
        trigtab[n*2] = (float)(cos(ang) * irho);
        trigtab[n*2+1] = (float)(sin(ang) * irho);
    }
    ttab = &trigtab[0]; //��ֵ�׵�ַ
    mdata0 = mask.data; 
    
    //��ʼд�����У�����д״̬
    cvStartWriteSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage, &writer );

    // stage 1. collect non-zero image points
    // �ռ�ͼ���е����з���㣬��Ϊ����ͼ���Ǳ�Եͼ�����Է������Ǳ�Ե��
    for( pt.y = 0, count = 0; pt.y < height; pt.y++ ) {
        //��ȡ����ͼ�����������ÿ�е�ַָ��
        //CvMat�ڲ���һ��������,��data��data����һ����Աptr����ʾ��ַ
        //CvMat����һ����Աstep,��ʾ�����ݳ��ȣ��ֽڣ�
        const uchar* data = image->data.ptr + pt.y * image->step;
        uchar* mdata = mdata0 + pt.y * width;

        for( pt.x = 0; pt.x < width; pt.x++ ) {
            if( data[pt.x] ){ //�Ǳ�Ե��
                mdata[pt.x] = (uchar)1; //���������Ӧλ������Ϊ1
                CV_WRITE_SEQ_ELEM( pt, writer ); //������д������
            }
            else //���Ǳ�Ե��
                mdata[pt.x] = 0; //���������Ӧλ������Ϊ0
        }
    }

    //д����ֻ����ִ��cvEndWriteSeq�����󣬲�����д�������У�֮ǰ�����ڻ�����
    seq = cvEndWriteSeq( &writer ); //��ֹд����
    count = seq->total; //�õ���Ե�������

    // stage 2. process all the points in random order
    // ����������еı�Ե��
    for( ; count > 0; count-- ) {
        // choose random point out of the remaining ones
        // ����1.��ʣ�µı�Ե�������ѡһ����,idxΪ������count�������
        int idx = cvRandInt(&rng) % count;
        int max_val = threshold-1, max_n = 0; //max_valΪ�ۼ��������ֵ,max_nΪ���ֵ����Ӧ�ĽǶ�
        //cvGetSeqElem �������������������
        CvPoint* point = (CvPoint*)cvGetSeqElem( seq, idx ); //�������idx����������ȡ�����������
        CvPoint line_end[2] = {{0,0}, {0,0}}; //����ֱ�ߵ������˵�
        float a, b;
        int* adata = (int*)accum.data; //�ۼ����ĵ�ַ��Ҳ����Hough�ռ�ĵ�ַָ��
        int i, j, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        i = point->y; //��������ĺ�������
        j = point->x;

        // "remove" it by overriding it with the last element
        // �������е����һ��Ԫ�ظ��ǵ��ղ��������������
        *point = *(CvPoint*)cvGetSeqElem( seq, count-1 );

        // check if it has been excluded already (i.e. belongs to some other line)
        // �������������Ƿ��Ѿ��������Ҳ�������Ѿ���������ֱ��
        // �����������㣬������������еĶ�Ӧλ������
        if( !mdata0[i*width + j] ) //��������Ѿ��������
            continue; 

        // update accumulator, find the most probable line
        // ����2.�����ۼ��������ҵ����п��ܵ�ֱ��
        for( n = 0; n < numangle; n++, adata += numrho ) {
            // �ɽǶȼ������
            // �� = xcos�� + ysin��
            r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
            r += (numrho - 1) / 2;
            int val = ++adata[r]; //���ۼ����������Ӧλ���ϼ�1������ֵ��val
            if( max_val < val ) { //�������ֵ�����õ����ĽǶ�
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        // ����3. ������沽��õ������ֵС����ֵ��������õ㣬��������
        if( max_val < threshold )
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        // ����4. �ӵ�ǰ�����������������ֱ�ߵķ���ǰ����ֱ������˵�Ϊֹ
        a = -ttab[max_n*2+1]; // a = -sin��
        b = ttab[max_n*2]; // b = cos��
        x0 = j; //��ǰ��ĺ�������ֵ
        y0 = i;

        //ȷ����ǰ������ֱ�ߵĽǶ�����45��-135��֮�䣬������0��-45���135��-180��֮��
        if( fabs(a) > fabs(b) ) {//��45��-135��֮��
            xflag = 1; //�ñ�־λ����ʶֱ�ߵĴ��Է���
            dx0 = a > 0 ? 1 : -1; //ȷ�����������λ����
            dy0 = cvRound( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1)); //ȷ��������
        }else {//��0��-45���135��-180��֮��
            xflag = 0;
            dy0 = b > 0 ? 1 : -1; //ȷ�����������λ����
            dx0 = cvRound( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1)); //ȷ��������
        }

        //����ֱ�ߵ������˵�
        for( k = 0; k < 2; k++ ) {
            //gap��ʾ����ֱ�ߵļ�϶, x��yΪ����λ�ã�dx��dyΪλ����  
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 ) //�����ڶ����˵��ʱ�򣬷�����λ��  
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            //����ֱ�ߵķ���λ�ƣ�ֱ������ͼ��ı߽���ļ�϶Ϊֹ  
            for( ;; x += dx, y += dy ) {
                uchar* mdata;
                int i1, j1;

                if( xflag ) {//ȷ���µ�λ�ƺ������λ��  
                    j1 = x;
                    i1 = y >> shift;
                }
                else {
                    j1 = x >> shift;
                    i1 = y;
                }

                //���������ͼ��ı߽磬ֹͣλ�ƣ��˳�ѭ��  
                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mdata0 + i1*width + j1;//��λλ�ƺ��������λ��  

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata ) { //�����벻Ϊ0��˵���õ��������ֱ����  
                    gap = 0; //���ü�϶Ϊ0  
                    line_end[k].y = i1;  //����ֱ�ߵĶ˵�λ��  
                    line_end[k].x = j1;
                }
                //����Ϊ0��˵������ֱ�ߣ����Լ���λ�ƣ�ֱ����϶���������õ���ֵΪֹ  
                else if( ++gap > lineGap )  //��϶��1  
                    break;
            }
        }

        //����5���ɼ�⵽��ֱ�ߵ������˵���Լ���ֱ�ߵĳ���  
        //��ֱ�߳��ȴ��������õ���ֵʱ��good_lineΪ1������Ϊ0  

        good_line = abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    abs(line_end[1].y - line_end[0].y) >= lineLength;

        //�ٴ������˵㣬Ŀ���Ǹ����ۼ�������͸�����������Ա���һ��ѭ��ʹ��  
        for( k = 0; k < 2; k++ ) {

            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy ) {
                uchar* mdata;
                int i1, j1;

                if( xflag ) {
                    j1 = x;
                    i1 = y >> shift;
                }
                else {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata ) {
                    //if���������������Щ�Ѿ��ж��Ǻõ�ֱ���ϵĵ��Ӧ���ۼ�����ֵ�������ٴ�������Щ�ۼ�ֵ  
                    if( good_line ) {
                        adata = (int*)accum.data; //�õ��ۼ��������ַָ��  
                        for( n = 0; n < numangle; n++, adata += numrho ) {
                            r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                            r += (numrho - 1) / 2;
                            adata[r]--; //��Ӧ���ۼ�����1  
                        }
                    }
                    //��������λ�ã������Ǻõ�ֱ�ߣ����ǻ���ֱ�ߣ�������Ӧλ�ö���0��
                    //�����´ξͲ������ظ�������Щλ���ˣ��Ӷ��ﵽ��С�����Ե���Ŀ��  
                    *mdata = 0;
                }
                //����Ѿ�������ֱ�ߵĶ˵㣬���˳�ѭ��  
                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }

        if( good_line ) {//����Ǻõ�ֱ�� 
            CvRect lr = { line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y };
            cvSeqPush( lines, &lr ); //�������˵�ѹ��������  
            if( lines->total >= linesMax )//�����⵽��ֱ������������ֵ�����˳��ú���  
                return;
        }
    }
}

/* Wrapper function for standard hough transform */
CV_IMPL CvSeq*
cvHoughLines2( CvArr* src_image, void* lineStorage, int method,
               double rho, double theta, int threshold,
               double param1, double param2 ) {
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image; //Դͼ��
    CvMat* mat = 0;
    CvSeq* lines = 0;
    CvSeq lines_header;
    CvSeqBlock lines_block;
    int lineType, elemSize;
    int linesMax = INT_MAX; //������ֱ�ߵ���������Ϊ�����
    int iparam1, iparam2;

    img = cvGetMat( img, &stub );

    if(!CV_IS_MASK_ARR(img)) //ȷ������ͼ����8λ��ͨ��
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if(!lineStorage) 
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if(rho <= 0 || theta <= 0 || threshold <= 0)
        CV_Error( CV_StsOutOfRange, "rho, theta and threshold must be positive" );


    lineType = CV_32SC4;
    elemSize = sizeof(int)*4;
    

    if( CV_IS_STORAGE( lineStorage )) {
        lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage );
    }
    else if( CV_IS_MAT( lineStorage )) {
        mat = (CvMat*)lineStorage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        if( CV_MAT_TYPE( mat->type ) != lineType )
            CV_Error( CV_StsBadArg,
            "The destination matrix data type is inappropriate, see the manual" );

        lines = cvMakeSeqHeaderForArray( lineType, sizeof(CvSeq), elemSize, mat->data.ptr,
                                         mat->rows + mat->cols - 1, &lines_header, &lines_block );
        linesMax = lines->total;
        cvClearSeq( lines );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    iparam1 = cvRound(param1);
    iparam2 = cvRound(param2);

    icvHoughLinesProbabilistic( img, (float)rho, (float)theta,
                threshold, iparam1, iparam2, lines, linesMax );

    if(mat){
        if(mat->cols > mat->rows)
            mat->cols = lines->total;
        else
            mat->rows = lines->total;
    }
    else
        result = lines;

    return result;
}


namespace cv
{

const int STORAGE_SIZE = 1 << 12;

static void seqToMat(const CvSeq* seq, OutputArray _arr)
{
    if( seq && seq->total > 0 )
    {
        _arr.create(1, seq->total, seq->flags, -1, true);
        Mat arr = _arr.getMat();
        cvCvtSeqToArray(seq, arr.data);
    }
    else
        _arr.release();
}

}

/*
 ����Hough�任����ֱ��ɨ�����еĵ㣬�������ѡ��һЩ�㣬
 ��ȷ��һ��ֱ�ߺ󣬽���ֱ���ϵ�δɨ���ֱ�Ӵ�ɨ���б���
 ȥ��
*/
void myHoughLinesP( InputArray _image, OutputArray _lines,
                      double rho, double theta, int threshold,
                      double minLineLength, double maxGap )
{
    Ptr<CvMemStorage> storage = cvCreateMemStorage(STORAGE_SIZE);//�����ڴ�洢��
    Mat image = _image.getMat();
    CvMat c_image = image;
    CvSeq* seq = cvHoughLines2( &c_image, storage, CV_HOUGH_PROBABILISTIC,
                    rho, theta, threshold, minLineLength, maxGap );
    seqToMat(seq, _lines); //����ת��ΪMat
}


/* End of file. */
