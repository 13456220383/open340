#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/*
def myGradX(src, dist) :
	h = src.shape[0]
	w = src.shape[1]
	roi_1 = src[:, dist : w]
	roi_2 = src[:, 0 : w - dist]
	out = np.zeros_like(src)
	half_dist = int(dist / 2)
	diff = cv2.absdiff(roi_1, roi_2)
	out[:, half_dist : w - dist + half_dist] = diff
	return out
*/

Mat myGradX(Mat src0, int dist)
{
	Mat src;
	cvtColor(src0, src, COLOR_BGR2GRAY);
	int h = src.rows;
	int w = src.cols;
	Mat roi_1 = src(Rect(dist, 0, w - dist, h));
	Mat roi_2 = src(Rect(0, 0, w - dist, h));
	Mat out = Mat::zeros(h, w, CV_8UC1);
	int half_dist = int(dist / 2);
	Mat diff;

	absdiff(roi_1, roi_2, diff);
	cout << "diff type" << diff.type() << endl;
	Rect roiRect = Rect(half_dist, 0, w - dist, h);
	//Mat ROI = out(Rect(half_dist, 0, w - dist, h));
	//out_obj = diff;
	Mat final;  // C++ 如何实现粘贴？？
	//add(out(Rect(half_dist, 0, w - dist, h)), out_obj, final);
	//add(out, out, final);
	cout << "out(roiRect) type" << out(roiRect).type() << endl;
	diff.copyTo(out(roiRect));
	//diff.convertTo(out(roiRect), out.type(), 1, 0);

	cout << "diff.rows" << diff.rows << endl;
	cout << "diff.cols" << diff.cols << endl;
	//通道一样就可以
	cout << "XXROI.rows" << out(roiRect).rows << endl;
	cout << "XXROI.cols" << out(roiRect).cols << endl;


	return out;
}

/*
def myGradY(src, dist) :
	h = src.shape[0]
	w = src.shape[1]
	roi_1 = src[dist:h, : ]
	roi_2 = src[0:h - dist, : ]
	out = np.zeros_like(src)
	half_dist = int(dist / 2)
	diff = cv2.absdiff(roi_1, roi_2)
	out[half_dist:h - dist + half_dist, : ] = diff
	return out
*/
Mat myGradY(Mat src0, int dist)
{
	Mat src;
	cvtColor(src0, src, COLOR_BGR2GRAY);
	int h = src.rows;
	int w = src.cols;
	Mat roi_1 = src(Rect(0, dist, w, h-dist));
	Mat roi_2 = src(Rect(0, 0, w, h-dist));
	Mat out = Mat::zeros(h, w, CV_8UC1);
	int half_dist = int(dist / 2);
	Mat diff;
	absdiff(roi_1, roi_2, diff);
	Mat ROI = out(Rect(0, half_dist, w, h - dist));
	diff.copyTo(ROI);
	cout << "YYROI.rows" << ROI.rows << endl;
	cout << "YYROI.cols" << ROI.cols << endl;
	//imshow("outy", out);
	//waitKey();
	return out;
}

/*
# 计算梯度的幅值
# src--原图
# x_dist--自差X方向平移的距离
# y_dist--自差Y方向平移的距离
@staticmethod
def myGradMag(src, x_dist, y_dist) :
	grad_x = ImgProc.myGradX(src, x_dist)  # 半图移动 半图的差
	grad_y = ImgProc.myGradY(src, y_dist)
	grad_x = grad_x.astype(np.float)
	grad_y = grad_y.astype(np.float)
	mag = cv2.magnitude(grad_x, grad_y)
	mag = mag.astype(np.uint8)
	# cv2.imshow("mag", mag)
	# cv2.waitKey(0)
	return mag
*/
Mat myGradMag(Mat src, int x_dist, int y_dist)
{
	int h = src.rows;
	int w = src.cols;
	Mat grad_x = myGradX(src, x_dist);
	Mat grad_y = myGradY(src, y_dist);
	cout << grad_x.cols << endl;
	cout << grad_x.rows << endl;

	cout << grad_y.cols << endl;
	cout << grad_y.rows << endl;
	grad_x.convertTo(grad_x, CV_32FC1, 1 / 255.0);
	grad_y.convertTo(grad_y, CV_32FC1, 1 / 255.0);
	//imshow("grad_x", grad_x);

	Mat mag = grad_x.clone();
	cout << grad_x.cols << endl;
	cout << grad_x.rows << endl;
	cout << grad_y.rows << endl;
	cout << grad_y.rows << endl;
	// 尺寸不一样 导致出错
	magnitude(grad_x, grad_y, mag);
	//imshow("mag", mag);
	//waitKey(0);
	mag.convertTo(mag, CV_8UC1,255);
	return mag;
}

Mat grad_diff(Mat src,Mat src2)
{
	Mat res_mag = myGradMag(src, 3, 3);
	Mat res_mag2 = myGradMag(src2, 3, 3);

	Mat diff;
	absdiff(res_mag, res_mag2, diff);
	return diff;
}

bool diff_degree_judge(Mat diff, double threash)
{
	Mat mat_mean, mat_stddev;
	//求均值
	meanStdDev(diff, mat_mean, mat_stddev);
	double m, s;
	m = mat_mean.at<double>(0, 0);
	//s = mat_stddev.at<double>(0, 0);
	cout << "mat_mean size" << mat_mean.size() << endl; //1*1
	cout << "diff size" << diff.size() << endl;  //1280*960
	cout << "m" << m << endl;
	if (m > threash)
		return true;
	else
		return false;

}


int main()
{
    Mat src = imread("/hj/1.jpg");
	Mat src2 = imread("/hj/2.jpg");
	Mat diff = grad_diff(src, src2);
	Mat mat_mean, mat_stddev;
	//求均值
	meanStdDev(diff, mat_mean, mat_stddev);
	double m, s;
	m = mat_mean.at<double>(0, 0);
	//s = mat_stddev.at<double>(0, 0);
	cout << "mat_mean size" << mat_mean.size() << endl; //1*1
	cout << "src size" << src.size() << endl;  //1280*960
	cout << "m" << m << endl;
	//imshow("kan", src);
	//Mat res = myGradY(src, 3);
	//Mat res_mag = myGradMag(src,3,3);
	//Mat res_mag2 = myGradMag(src2, 3, 3);
	//imshow("g", res);
	//imshow("res_mag", res_mag);
	//imshow("res_mag2", res_mag2);
	//Mat diff;
	//absdiff(res_mag, res_mag2, diff);
	bool x = diff_degree_judge(diff, 3.0);
	cout << "x=" << x << endl;
	imwrite("/hj/diff.jpg", diff);
	//waitKey(0);
}



// 编译命令
//arm-linux-gnueabihf-g++ -I/usr/local/arm/lib/opencv340/include -L/usr/local/arm/lib/opencv340/lib
//-lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_highgui -lpthread -lrt -ldl -o gradDiff_arm_code
//gradDiff_arm_code.cpp
