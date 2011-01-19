//============================================================================================//
//    The cvlibfreenect-System-Project                                                   
//    by                                                                    
//    Rytta Communications Inc.                                           
//    (c) Version 0.21. January 2010 
//
//    EXPERIMENTAL -> compile opencv & libfreenect                                 
//                                                                    
//===========================================================================================// 
//     cvOpenKinectDepthTrack21.cpp   
//     ---------------------------    
//     
//     begin     : Fri. 14. January 12.15:00 GMT 2010                            
//     copyright : (C) 2008/2009/2010 by  s.morf 
//     email     : stevenmorf@bluewin.ch 
//     originalCompile with   
//     opencv & openkinect  
//     g++ -fPIC -g -Wall -I/usr/local/include/opencv -L /usr/local/lib -lcxcore -lcv -lhighgui -lcvaux -lml -o cvOpenKinectDepthTrack21 cvOpenKinectDepthTrack21.cpp -lfreenect -lm 

//     run as    : ./cvOpenKinectDepthTrack21
//
//    for: 
//    Basics of : - interduction to compile opencv with libfreenect
//                - basics for grayDepthTrack
//                                                   
//
/*************************************************************************************************
References:

************************************************************************************************** 
     THIS PROGRAMM IS UNDER THE GNU-LICENCE                              
*************************************************************************************************  
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA -02111-1307  USA

*************************************************************************************************
NOTES:
13.01. - interductions of Grayscale DepthBodyTracking
         - 1.compile (libs) 13.01.2011
**************************************************************************************************************/
#include "libfreenect.hpp"
#include <iostream> 
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>
#include <string.h>

using namespace cv;
using namespace std;

//Some defines we left out of the book
#define CVX_RED   CV_RGB(0xff,0x00,0x00)
#define CVX_GREEN CV_RGB(0x00,0xff,0x00)
#define CVX_BLUE  CV_RGB(0x00,0x00,0xff)

int I,J,K;    
int HEIGHT,WIDTH,STEP,CHANNELS; 
int STEPR, CHANNELSR;           
int TEMP;                       

uchar *DATA,*DATAR;            
CvCapture *CAPTURE;             
IplImage *FRAME;               
IplImage *RESULT1;              

bool quit;
bool done;

int key = 0;

int XP,YP;
CvPoint PT1,PT2;   //pt1,pt2;
CvPoint* PT[2];    //pt[2];

int LEVELS = 3;    // levels = 3;
CvSeq* CONTOURS = 0;   // *cocontours = 0;


int cvWorkCounter;

// bodyContours
IplImage* IMG_8UC1;     
       
IplImage* IMG_EDGE; 
IplImage* IMG_8UC3;
;
CvMemStorage* STORAGE1;
CvSeq* FIRST_CONTOUR;
CvSeq* C;
      
int NC; 
int N = 0,KK;

// myBox1)
struct points
  {
     int x;
     int y;
  } BODYBOX;  

// Declare font
CvFont myFont;

CvSeq *A; 
CvSize SZ;

IplImage *SRC;
IplImage *HSV_IMAGE;
IplImage *HSV_MASK;
IplImage *HSV_EDGE;

IplImage *CONTOUR;
      
CvScalar HSV_MIN;
CvScalar HSV_MAX;
     
CvMemStorage *STORAGE;
CvMemStorage *AREASTORAGE;
CvMemStorage *MINSTORAGE;
CvMemStorage *DFTSTORAGE;

// CvSeq *contours = NULL;
CvSeq *CONTOURS2 = NULL;

IplImage *BG;
CvRect RECT;

CvSeq *HULL;
CvSeq *DEFECT;

CvBox2D BOX;

double RESULT, RESULT2; 
int CHECKCXT;

int DEF[10];
float TDEF;
int IDEF;

// NEW STUFF FROM HeadTrackCommunications4.cpp
// target
int BCENTERX = 320;
int BCENTERY = 240;

int BCAMX = BCENTERX;
int BCAMY = BCENTERY;

// myBox
struct TPOINTS
  {
     int x1;
     int y1;
     int x2;
     int y2; 
     int width;
     int height; 
  } TARGETBOX;  

// myBox1)
struct POINTS
  {
     int x;
     int y;
  } BODYDBOX;  

bool BTRACKINIT   = false;
bool BOBENLINKS   = false;
bool BOBENRECHTS  = false;
bool BUNTENRECHTS = false;
bool BUNTENLINKS  = false;

bool BOOLBODYTARGET = false; 

bool BODYDONEOLX = false;
bool BODYDONEOLY = false;
bool BODYDONEULX = false;
bool BODYDONEULY = false;
bool BODYDONEORX = false;
bool BODYDONEORY = false;
bool BODYDONEURX = false;
bool BODYDONEURY = false;

bool BODYDONEX = false;
bool BODYDONEY = false;

int rc = 29, gc = 29, bc = 29;
     
// dirty stuff for esc=exit
pthread_t freenect_thread;

IplImage* greyscale;
IplImage* dst;

/************************************************************************************************************/
class Mutex {
public:
	Mutex() {
		pthread_mutex_init( &m_mutex, NULL );
	}
	void lock() {
		pthread_mutex_lock( &m_mutex );
	}
	void unlock() {
		pthread_mutex_unlock( &m_mutex );
	}
private:
	pthread_mutex_t m_mutex;
};
/************************************************************************************************************/
class MyFreenectDevice : public Freenect::FreenectDevice {
  public:
	MyFreenectDevice(freenect_context *_ctx, int _index)
		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT_SIZE),m_buffer_rgb(FREENECT_VIDEO_RGB_SIZE), m_gamma(2048), m_new_rgb_frame(false), m_new_depth_frame(false),
		  depthMat(Size(640,480),CV_16UC1), rgbMat(Size(640,480),CV_8UC3,Scalar(0)), ownMat(Size(640,480),CV_8UC3,Scalar(0))
	{
		for( unsigned int i = 0 ; i < 2048 ; i++) {
			float v = i/2048.0;
			v = std::pow(v, 3)* 6;
			m_gamma[i] = v*6*256;
		}
	}
	// Do not call directly even in child
	void VideoCallback(void* _rgb, uint32_t timestamp) {
		std::cout << "RGB callback" << std::endl;
		m_rgb_mutex.lock();
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		rgbMat.data = rgb;
		m_new_rgb_frame = true;
		m_rgb_mutex.unlock();
	};
	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {
		std::cout << "Depth callback" << std::endl;
		m_depth_mutex.lock();
		uint16_t* depth = static_cast<uint16_t*>(_depth);
		depthMat.data = (uchar*) depth;
		m_new_depth_frame = true;
		m_depth_mutex.unlock();
	}

	bool getVideo(Mat& output) {
		m_rgb_mutex.lock();
		if(m_new_rgb_frame) {
			cv::cvtColor(rgbMat, output, CV_RGB2BGR);
			m_new_rgb_frame = false;
			m_rgb_mutex.unlock();
			return true;
		} else {
			m_rgb_mutex.unlock();
			return false;
		}
	}

	bool getDepth(Mat& output) {
			m_depth_mutex.lock();
			if(m_new_depth_frame) {
				depthMat.copyTo(output);
				m_new_depth_frame = false;
				m_depth_mutex.unlock();
				return true;
			} else {
				m_depth_mutex.unlock();
				return false;
			}
		}

  private:
  // new stuff boo-stuff before Mat-stuff no warnings compile and run OK
   std::vector<uint8_t> m_buffer_depth;
	std::vector<uint8_t> m_buffer_rgb;
	std::vector<uint16_t> m_gamma;

   bool m_new_rgb_frame;
	bool m_new_depth_frame;

	Mat depthMat;
	Mat rgbMat;
	Mat ownMat;

Mat depthMat1;
Mat depthf1;
	Mutex m_rgb_mutex;
	Mutex m_depth_mutex;
	
};
/***********************************************************************************************************************************************************/
void sum_rgb( IplImage* src, IplImage* dst ) 
  {
     // Allocate individual image planes.
     IplImage* r = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
     IplImage* g = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
     IplImage* b = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
     
     // Temporary storage.
     IplImage* s = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
      
     // Split image onto the color planes.
     cvSplit( src, r, g, b, NULL );
     
     // Add equally weighted rgb values.
     cvAddWeighted( r, 1./3., g, 1./3., 0.0, s );
     cvAddWeighted( s, 2./3., b, 1./3., 0.0, s );
     
     // Truncate values above 100. 75
     cvThreshold( s, dst, 75, 255, CV_THRESH_TRUNC ); //_BINARY | CV_THRESH_OTSU );
     // 15,150.....
     cvThreshold( dst, dst, 15, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );

     cvSaveImage("bw1.jpg",dst);
     
     cvReleaseImage( &r );
     cvReleaseImage( &g );   
     cvReleaseImage( &b );   
     cvReleaseImage( &s );
  
  }
/***********************************************************************************************************************************************************/
int DepthGrayBodyTrack()
  {
     IplImage* gray = cvLoadImage("gray.jpg",1);
     greyscale = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);
            
     // manually convert to greyscale
     for(int y = 0; y < gray->height; y++) 
        {
           uchar* p = (uchar*) gray->imageData + y* gray->widthStep; // pointer to row
           uchar* gp = (uchar*) greyscale->imageData + y*greyscale->widthStep;  
                 
           for(int x = 0; x < gray->width; x++)
              {
                 gp[x] = (p[3*x] + p[3*x+1] + p[3*x+2])/3;   // average RGB values 

              }
        }
                                                                                       
     cvNamedWindow("GRAY",CV_WINDOW_AUTOSIZE); 
     cvMoveWindow("GRAY", 0, 480 );
           
     cv::imshow("GRAY",greyscale);  
     cv::imwrite("gray1.jpg",greyscale);

     IplImage* src = cvLoadImage( "gray1.jpg", 1 );
     dst = cvCreateImage( cvGetSize(src), src->depth, 1);
    
     sum_rgb( src, dst);

     cv::imshow("GRAY",dst);
     cv::imwrite("gray2.jpg",dst); 

     return 0;
     
  }
/***********************************************************************************************************************************************************/
int main(int argc, char **argv) 
  {
	  bool die(false);

	  Mat depthMat(Size(640,480),CV_16UC1);
	  Mat depthf  (Size(640,480),CV_8UC1);
 	  Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
	  Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));
 
	  Freenect::Freenect<MyFreenectDevice> freenect;
	  MyFreenectDevice& device = freenect.createDevice(0);
      
     //new  windowScreenPosition ok
     cvNamedWindow("Main",CV_WINDOW_AUTOSIZE);
     cvMoveWindow("Main", 0, 0);   

	  cvNamedWindow("Depth",CV_WINDOW_AUTOSIZE); 
     cvMoveWindow("Depth", 700, 0 );

     device.startVideo();
	  device.startDepth();
      
     while(!die) 
        {
    	   device.getVideo(rgbMat);
   	   device.getDepth(depthMat);
         
           // original         
           cv::imshow("Main", rgbMat);
           
           // save output for colorDetections()
           cv::imwrite("original.jpg",rgbMat);
                      
           // lesen
           cv::Mat img1 = cv::imread("original.jpg",1);
           
           // zuweisung iplImage Frame for Colordetections
           FRAME = cvLoadImage("original.jpg", 1);

           depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
           cv::imwrite("gray.jpg",depthf);

           cv::imshow("Depth",depthf);

           DepthGrayBodyTrack();

           key = cvWaitKey(10) & 0xFF;

           if(key == 27) // ESC 
              {
                 die = true;
                 cvDestroyWindow("Main");
		           cvDestroyWindow("Depth"); 

                 // here zwischen loesung fuer esc == exit
                 pthread_join(freenect_thread, NULL);
	              pthread_exit(NULL);
                 break;
              }
 
        }

     device.stopVideo();
     device.stopDepth();
     
     return 0;
  }
/**********************************************************************************************************************************************************/
