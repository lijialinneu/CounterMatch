#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <exception>
using namespace cv;
using namespace std;

/*
 * 卷积计算，求梯度
 */
void filter(Mat src, Mat dst, float k[9]) {
    Mat kernel = Mat(3, 3, CV_32FC1, k);
    filter2D(src, dst, CV_16S, kernel, Point(-1,-1), 0.0, 4);
}

void myCanny( InputArray _src, OutputArray _dst, 
              double low_thresh, double high_thresh,
              int aperture_size) {

    Mat src = _src.getMat();
    CV_Assert( src.depth() == CV_8U );
    // 扩大src的边界，利于后续5x5模板运算
    // copyMakeBorder(src, src, 2, 2, 2, 2, IPL_BORDER_REPLICATE);
    _dst.create(src.size(), CV_8U);
    Mat dst = _dst.getMat();

    if (low_thresh > high_thresh) 
        std::swap(low_thresh, high_thresh);

    const int cn = src.channels();
    CV_Assert(cn == 1);

	//-----------------------------------------
	//  【1】 改进Scharr滤波器
	//  添加45°、135°、180°、225°、270°、315°方向的滤波器
	//-----------------------------------------
    //float k_x[9] = {-3,0,3,-10,0,10,-3,0,3};   //0
    //float k_y[9] = {-3,-10,-3,0,0,0,3,10,3};   //90
    //float k_45[9] = {0,3,10,-3,0,3,-10,-3,0};  //45
    //float k_135[9] = {10,3,0,3,0,-3,0,-3,-10}; //135
    //float k_180[9] = {3,0,-3,10,0,-10,3,0,-3}; //180
    //float k_225[9] = {0,-3,-10,3,0,-3,10,3,0}; //225
    //float k_270[9] = {3,10,3,0,0,0,-3,-10,-3}; //270
    //float k_315[9] = {-10,-3,0,-3,0,3,0,3,10}; //315


	float k_x[9] = {-1, 0, 1, -3, 0, 3, -1, 0, 1};   //0
    float k_y[9] = {-1, -3, -1, 0, 0, 0, 1, 3, 1};   //90
    float k_45[9] = {0, 1, 3, -1, 0, 1, -3, -1, 0};  //45
    float k_135[9] = {3, 1, 0, 1, 0, -1, 0, -1, -3}; //115

	//float k_x[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};   //0
 //   float k_y[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};   //90
 //   float k_45[9] = {0, 3, 10, -3, 0, 3, -10, -3, 0};  //45
 //   float k_135[9] = {10, 3, 0, 3, 0, -3, 0, -3, -10}; //135
    

    Mat dx(src.rows, src.cols, CV_16SC(cn));
    Mat dy(src.rows, src.cols, CV_16SC(cn));
    Mat d_45(src.rows, src.cols, CV_16SC(cn));
    Mat d_135(src.rows, src.cols, CV_16SC(cn));
    //Mat d_180(src.rows, src.cols, CV_16SC(cn));
    //Mat d_225(src.rows, src.cols, CV_16SC(cn));
    //Mat d_270(src.rows, src.cols, CV_16SC(cn));
    //Mat d_315(src.rows, src.cols, CV_16SC(cn));

	// 计算x方向的sobel方向导数，计算结果存在dx中 
	//Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);   
    // 计算y方向的sobel方向导数，计算结果存在dy中  

    filter(src, dx, k_x);
    filter(src, dy, k_y);
    filter(src, d_45, k_45);
    filter(src, d_135, k_135);
    //filter(src, d_180, k_180);
    //filter(src, d_225, k_225);
    //filter(src, d_270, k_270);
    //filter(src, d_315, k_315);

    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);

    ptrdiff_t mapstep = src.cols + 4;

    // 由3改为5
    AutoBuffer<uchar> buffer((src.cols + 4) * (src.rows + 4) + cn * mapstep * 5 * sizeof(int));  

    int* mag_buf[5];
    mag_buf[0] = (int*)(uchar*)buffer;       
    mag_buf[1] = mag_buf[0] + mapstep*cn;    
    mag_buf[2] = mag_buf[1] + mapstep*cn;    
    mag_buf[3] = mag_buf[2] + mapstep*cn;
    mag_buf[4] = mag_buf[3] + mapstep*cn;

    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));  

    uchar* map = (uchar*)(mag_buf[4] + mapstep*cn);   // 2->4
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep); 

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);   
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++){
        //int* _norm = mag_buf[(i > 0) + 1] + 1;   
        int* _norm = mag_buf[(i > 0) + 1] + 1;   

        if (i < src.rows) {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);
            short* _d45 = d_45.ptr<short>(i);
            short* _d135 = d_135.ptr<short>(i);
            //short* _d180 = d_180.ptr<short>(i);
            //short* _d225 = d_225.ptr<short>(i);
            //short* _d270 = d_270.ptr<short>(i);
            //short* _d315 = d_315.ptr<short>(i);

            for (int j = 0; j < src.cols*cn; j++) {
                //_norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
				// 2017 05 25
	/*			_norm[j] = (
                    abs(int(_dx[j])) + abs(int(_dy[j])) + 
                    abs(int(_d45[j])) + abs(int(_d135[j])) +
                    abs(int(_d180[j])) + abs(int(_d225[j])) +
                    abs(int(_d270[j])) + abs(int(_d315[j]))
                    ) / 4;*/
				int p45 =  abs(int(_d45[j]));
				int p135 =  abs(int(_d135[j]));
				int px = abs(int(_dx[j])) + (p45 + p135) / 2;
				int py = abs(int(_dy[j])) + (p45 - p135) / 2;
				_norm[j] = px + py;

            }

            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep * i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[2] + 1; // take the central row 1改为2
        /*ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];*/
        ptrdiff_t magstep0 = mag_buf[0] - mag_buf[2];
        ptrdiff_t magstep1 = mag_buf[1] - mag_buf[2];
        ptrdiff_t magstep3 = mag_buf[3] - mag_buf[2];
        ptrdiff_t magstep4 = mag_buf[4] - mag_buf[2];

        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);

        if ((stack_top - stack_bottom) + src.cols > maxsize) { // 分配栈空间
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;

        for (int j = 0; j < src.cols; j++) {
			//-----------------------------------------
			//  【2】 自动调整阈值
			//  当背景很弱，阈值适当降低
			//  当背景很强，阈值适当增加
			//-----------------------------------------
           
			int sum = 0;
			for(int p = j - 1;j >= 1 && p <= j + 1; p++) {
                sum += (_mag[p + magstep0] + _mag[p + magstep1] + _mag[p]) / 3;          
		    }
			int average = sum / 3;
			if(average < cvFloor(low_thresh)) {
				low = cvFloor(low_thresh) - average / 3;
				high = cvFloor(high_thresh) - average / 3;
			}else {			
				low = cvFloor(low_thresh) + average / 3;
				high = cvFloor(high_thresh) + average / 3;
			}

            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5);
            int m = _mag[j];
            if (m > low) { // 如果大于低阈值            
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;
                int tg22x = x * TG22;
                if (y < tg22x){                    
                    if (m >= _mag[j - 1] && m >= _mag[j + 1] && 
						m >= _mag[j - 2] && m >= _mag[j + 2]) {
                        goto __ocv_canny_push;
                    }                        
                }else {
                    int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if (y > tg67x) {                        
                        if (m >= _mag[j + magstep1] && m >= _mag[j + magstep3] && 
							m >= _mag[j + magstep0] && m >= _mag[j + magstep4]) {
							goto __ocv_canny_push;
                        }
                    }else {
                        int s = (xs ^ ys) < 0 ? -1 : 1;                        
                        if (m >= _mag[j + magstep1 - s] && m >= _mag[j + magstep3 + s] && 
							m >= _mag[j + magstep0 - 2 * s] && m >= _mag[j + magstep4 + 2 * s]) {
							goto __ocv_canny_push;
                        }
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
__ocv_canny_push:
            if (!prev_flag && m > high && _map[j-mapstep] != 2) {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }
        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        // mag_buf[2] = _mag;
        mag_buf[2] = mag_buf[3];
        mag_buf[3] = mag_buf[4];
        mag_buf[4] = _mag;
    }

	int s1 = 0, s2 = 0, s3 = 0, s4 = 0;

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom) {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize) {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }
        CANNY_POP(m);

		

		


		//-----------------------------------------
		//  【3】 边缘跟踪策略
		//  首先，定义一个3层前向浅层神经网络，
		//  用于计算5x5邻域内的边缘方向
		//-----------------------------------------

		
        ptrdiff_t twostep = mapstep + mapstep; 
        int a[5][5] = {
            { m[-twostep-2], m[-twostep-1], m[-twostep], m[-twostep+1], m[-twostep+2] }, 
            { m[-mapstep-2], m[-mapstep-1], m[-mapstep], m[-mapstep+1], m[-mapstep+2] }, 
            { m[-2],         m[-1],         m[0],        m[1],          m[2] }, 
            { m[mapstep-2],  m[mapstep-1],  m[mapstep],  m[mapstep+1],  m[mapstep+2] }, 
            { m[twostep-2],  m[twostep-1],  m[twostep],  m[twostep+1],  m[twostep+2] }
        };

		int count = 0;
		for(int i = 0; i < 5; i++) {
			for(int j = 0; j < 5; j++) {
			    if(a[i][j] == 0) count++;
			}
		}
		if(count > 10) continue;

		if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);


		/*
		int x[8];
        x[0] = a[2][0] + a[2][1] + a[2][3] + a[2][4]; // 0°方向
        x[1] = a[3][0] + a[3][1] + a[1][3] + a[1][4]; // 22.5°方向
        x[2] = a[4][0] + a[3][1] + a[1][3] + a[0][4]; // 45°方向
        x[3] = a[4][1] + a[3][1] + a[1][3] + a[0][3]; // 67.5°方向
		x[4] = a[4][2] + a[3][2] + a[1][2] + a[0][2]; // 90°方向
		x[5] = a[0][1] + a[1][1] + a[3][3] + a[4][3]; // 112.5°方向
		x[6] = a[0][0] + a[1][1] + a[3][3] + a[4][4]; // 135°方向
		x[7] = a[1][0] + a[1][1] + a[3][3] + a[3][4]; // 157.5°方向

		sort(x, x + 8);
		int min_val = x[0];
		int direction = 0;
		if(x[0] > 3) continue;
		
		for(int i = 0; i < 8; i++) {
			
				switch(i) {
				case 0:
					if(!a[2][0]) CANNY_PUSH(m - 2);
					if(!a[2][1]) CANNY_PUSH(m - 1);
					if(!a[2][3]) CANNY_PUSH(m + 1);
					if(!a[2][4]) CANNY_PUSH(m + 2);
					break;
				case 1:
					if(!a[3][0]) CANNY_PUSH(m + mapstep - 2);
					if(!a[3][1]) CANNY_PUSH(m + mapstep - 1);
					if(!a[1][3]) CANNY_PUSH(m - mapstep + 1);
					if(!a[1][4]) CANNY_PUSH(m - mapstep + 2);
					break;
				case 2:
					//if(!a[4][0]) CANNY_PUSH(m + twostep - 2 );
					if(!a[3][1]) CANNY_PUSH(m + mapstep - 1);
					if(!a[1][3]) CANNY_PUSH(m - mapstep + 1);
					//if(!a[0][4]) CANNY_PUSH(m - twostep + 2);
					break;
				case 3:
					if(!a[4][1]) CANNY_PUSH(m + twostep - 1);
					if(!a[3][1]) CANNY_PUSH(m + mapstep - 1);
					if(!a[1][3]) CANNY_PUSH(m - mapstep + 1);
					if(!a[0][3]) CANNY_PUSH(m - twostep + 1);
					break;
				case 4:
					//if(!a[0][2]) CANNY_PUSH(m - twostep); 
					if(!a[1][2]) CANNY_PUSH(m - mapstep);
					if(!a[3][2]) CANNY_PUSH(m + mapstep);
					if(!a[4][2]) CANNY_PUSH(m + twostep);
					break;
				case 5:
					if(!a[4][3]) CANNY_PUSH(m + twostep + 1);
					if(!a[3][3]) CANNY_PUSH(m + mapstep + 1);
					if(!a[1][1]) CANNY_PUSH(m - mapstep - 1);
					if(!a[0][1]) CANNY_PUSH(m - twostep - 1);
					break;
				case 6:
					if(!a[0][0]) CANNY_PUSH(m - twostep - 2);
					if(!a[1][1]) CANNY_PUSH(m - mapstep - 1);
					if(!a[3][3]) CANNY_PUSH(m + mapstep + 1);
					if(!a[4][4]) CANNY_PUSH(m + twostep + 2);
					break;
				case 7:
					if(!a[1][0]) CANNY_PUSH(m - mapstep - 2);
					if(!a[1][1]) CANNY_PUSH(m - mapstep - 1);
					if(!a[3][3]) CANNY_PUSH(m + mapstep + 1);
					if(!a[3][4]) CANNY_PUSH(m + mapstep + 2);
					break;
			    }
			
		}**/

    }


    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step) {
        for (int j = 0; j < src.cols; j++) {
            pdst[j] = (uchar)-(pmap[j] >> 1);
        }
    }
}


/* End of file. */
