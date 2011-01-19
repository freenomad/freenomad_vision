//============================================================================================//
//    The cvlibfreenect-System-Project                                                   
//    by                                                                    
//    Rytta Communications Inc.                                           
//    (c) Version 0.23. January 2010 
//
//    EXPERIMENTAL -> compile opencv & libfreenect                                 
//                                                                    
//===========================================================================================// 
//     cvOpenKinectDepthTrack23.cpp   
//     ---------------------------    
//     
//     begin     : Sat. 15. January 13.00:00 GMT 2010                            
//     copyright : (C) 2008/2009/2010 by  s.morf 
//     email     : stevenmorf@bluewin.ch 
//     originalCompile with   
//     opencv & openkinect  
//     g++ -fPIC -g -Wall -I/usr/local/include/opencv -L /usr/local/lib -lcxcore -lcv -lhighgui -lcvaux -lml -o cvOpenKinectDepthTrack23 cvOpenKinectDepthTrack23.cpp -lfreenect -lm 

//     run as    : ./cvOpenKinectDepthTrack23
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
15.01. - New Version 0.23. with  einbau: colorDetections()
         - sehr experimentell absturzgefaehrtet
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
int BodyContours()
  {
     cvNamedWindow("RED",CV_WINDOW_AUTOSIZE); 
     cvMoveWindow("RED", 700, 240 );
      
     IMG_8UC1 = cvLoadImage("bw2.jpg",-1);     
       
     IMG_EDGE = cvCreateImage( cvGetSize( IMG_8UC1 ), 8, 1 );
     IMG_8UC3 = cvCreateImage( cvGetSize( IMG_8UC1 ), 8, 3 );
     cvThreshold( IMG_8UC1, IMG_EDGE, 128, 255, CV_THRESH_BINARY );
     STORAGE1 = cvCreateMemStorage();
     CvSeq* FIRST_CONTOUR = NULL;

     cvFindContours( IMG_EDGE, STORAGE1, &FIRST_CONTOUR, sizeof(CvContour), CV_RETR_LIST );
     CONTOURS2 = NULL;
     RESULT = 0, RESULT2 = 0; 
     
     while(FIRST_CONTOUR)
        {
           RESULT = fabs( cvContourArea( FIRST_CONTOUR, CV_WHOLE_SEQ ) );
                       
           if( RESULT > RESULT2 ) 
              {
                 RESULT2 = RESULT; CONTOURS2 = FIRST_CONTOUR;
              }
           FIRST_CONTOUR  =  FIRST_CONTOUR->h_next;
        }
     if( CONTOURS2 )
        {
           RECT = cvBoundingRect( CONTOURS2, 0 );

           cvRectangle(IMG_8UC3,cvPoint(RECT.x,RECT.height), cvPoint(RECT.x + RECT.width, RECT.y), CV_RGB(200, 0, 200), 1, 8, 0 );
                    
           HULL = cvConvexHull2( CONTOURS2, 0, CV_CLOCKWISE, 0 );
           DEFECT = cvConvexityDefects( CONTOURS2, HULL, DFTSTORAGE );
/*
           IDEF++;  
           DEF[IDEF] = DEFECT->total;
           
           if(IDEF == 10)
              {
                 TDEF = (float)(DEF[1]+DEF[2]+DEF[3]+DEF[4]+DEF[5]+DEF[6]+DEF[7]+DEF[8]+DEF[9]+DEF[10]) / 10.0;
                 printf("\n\nTDEF: %f\n\n",TDEF);
                 IDEF = 1;
              }   
*/

           if( DEFECT->total >=40 ) 
              { 
                 cout << " Closed Palm " << endl;
              }
           else if( DEFECT->total >=30 && DEFECT->total <40 ) 
              {
                 cout << " Open Palm " << endl;
              }
           else
              { 
                 cout << " Fist " << endl;
              }

           cout << "Defects: " << DEFECT->total << endl;
           
           BOX = cvMinAreaRect2( CONTOURS2, MINSTORAGE );

           // erzeugt red HandContour                                                  // level 3
           cvDrawContours( IMG_8UC3, CONTOURS2, CV_RGB(255,0,0), CV_RGB(0,255,0), 2, 3, CV_AA, cvPoint(0,0) );
                                  
           cvCircle( IMG_8UC3, cvPoint(BOX.center.x, BOX.center.y), 3, CV_RGB(200, 0, 200), 2, 8, 0 ); 
           //      cvEllipse( IMG_8UC3, cvPoint(BOX.center.x, BOX.center.y), cvSize(BOX.size.height/2, BOX.size.width/2), BOX.angle, 0, 360, CV_RGB(220, 0, 220), 1, 8, 0 );

           cvEllipse( IMG_8UC3, cvPoint(BOX.center.x, BOX.center.y), cvSize(BOX.size.height/2, BOX.size.width/2), BOX.angle, 0, 360, cvScalar(0x00,0xff,0xff), 1, 8, 0 );
                    
           BODYBOX.x = BOX.center.x;
           BODYBOX.y = BOX.center.y;
           printf("BodyCenter X: %d, BodyCenter Y: %d\n",BODYBOX.x,BODYBOX.y); 
          
           // anzeige TargetBox (yellow Rect)

           TARGETBOX.x1 = BCENTERX - 20; 
           TARGETBOX.y1 = BCENTERY - 20;  
           TARGETBOX.x2 = BCENTERX + 20; 
           TARGETBOX.y2 = BCENTERY + 20; 
           TARGETBOX.width  = 40;
           TARGETBOX.height = 40;

    //      cvRectangle( IMG_8UC3, cvPoint( TARGETBOX.x1,TARGETBOX.y1 ), cvPoint( TARGETBOX.x2, TARGETBOX.y2 ), cvScalar(0x00,0xff,0xff),8,-1); // cvScalar(255,255,255), 8, -1 );  
           
           //BodyTrackCalc();
            
        } // end contour 2
      
     // original from cvColor     
     for( C = FIRST_CONTOUR; C !=NULL; C = C->h_next ) 
        { 
           cvWorkCounter++;
           
           cvCvtColor( IMG_8UC1, IMG_8UC3 , CV_GRAY2BGR );
           
           cvDrawContours( IMG_8UC3, C,
                           CVX_RED,  //Yarg, these are defined above, but not in the book.  Oops
                           CVX_BLUE,
                           0,        // Try different values of max_level, and see what happens
                           2,
                           8
                         );
           
           N++;
           printf("Contour #%d\n", N );
           
           // here das wichtige bild
           cvShowImage( "RED", IMG_8UC3 );
        
           printf(" %d elements:\n", C->total );

           // das funzzt noch nicht richtig
           for(int II = 0; II <= C->total; ++II) 
              {
                 CvPoint* P = CV_GET_SEQ_ELEM( CvPoint, C, II );
                       
                 // cvNamedWindow("FreeSurface", CV_WINDOW_AUTOSIZE );
                 // IplImage* freeSurfaceImg;         //160,120
                 // freeSurfaceImg = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U, 3); 
                             
                 // cvZero(freeSurfaceImg);
                                            
                 // here zuweisung bestcontour into points
                 XP = P->x; 
                 YP = P->y;
                             
                 PT1.x = XP;
                 PT1.y = YP;
                 PT2.x = XP;
                 PT2.y = YP; 

                 cvLine( RESULT1, PT1, PT2,CV_RGB(255,0,255),4 ,8 ); 
                 cvLine( RESULT1, PT1,PT2,CV_RGB(128,0,255),4 ,8 ); 

                                     
              } // end ForLoop 
           
           N++;
             
        } // end   for( CvSeq* c=first_contour; c!=NULL; c=c->h_next ) 
      
     cvShowImage( "RED", IMG_8UC3 );
     cvSaveImage( "red.jpg",IMG_8UC3 );

     N = 0;
     printf("DONE\n");

     return 0;
     
  }
/*****************************************************************************/
int colorDetections()
  {
     printf("ColorDetections\n");
       
     RESULT1 = cvCreateImage( cvGetSize( dst ), 8, 1 ); //dst

  
     HEIGHT = FRAME->height;
     WIDTH = FRAME->width;

     STEP = FRAME->widthStep;
     CHANNELS = FRAME->nChannels;
     DATA = (uchar *)FRAME->imageData;
            
     //  Here I use the Suffix r to diffenrentiate the result data and the frame data
     //  Example :stepr denotes widthStep of the result IplImage and step is for frame IplImage
               
     STEPR = RESULT1->widthStep;
     CHANNELSR = RESULT1->nChannels;
           
     DATAR = (uchar *)RESULT1->imageData;
           
     for(I = 0;I < (HEIGHT); I++) 
        {
           for(J = 0;J <(WIDTH); J++)
              {
                 //  As I told you previously you need to select pixels which are
                 //  more red than any other color
                 //  Hence i select a difference of 29(which again depends on the scene).
                 //  (you need to select randomly and test
                      
                 if(((DATA[I * STEP + J * CHANNELS + 2]) > (29 + DATA[I * STEP + J *CHANNELS]))
                      && ((DATA[I * STEP + J * CHANNELS + 2]) > (29 + DATA[I * STEP + J * CHANNELS+1])))
                    {
                       DATAR[I * STEPR + J * CHANNELSR] = 255;
                    }
                 else
                    {
                       DATAR[I * STEPR + J * CHANNELSR] = 0;
                    }
              }
        }

     return 0;
  }  

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

     // new 15.01.
     colorDetections();
     cvSaveImage("bw2.jpg",RESULT1);

     BodyContours();
        
     cvReleaseImage( &r );
     cvReleaseImage( &g );   
     cvReleaseImage( &b );   
     cvReleaseImage( &s );

    // return 0;
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
/****************************************************************************/
