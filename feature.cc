#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

// Function to detect chessboard bounding box using Shi-Tomasi corner detection
Rect detectChessboardBBox(Mat& frame, int minArea = 5000) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // Use Shi-Tomasi corner detection
    vector<Point2f> corners;
    goodFeaturesToTrack(gray, corners, 100, 0.01, 10);

    if (corners.empty() || corners.size() < 4) {
        return Rect();
    }

    // Create a mask with detected corners
    Mat thresh = Mat::zeros(gray.size(), CV_8UC1);
    for (auto &corner : corners) {
        circle(thresh, corner, 5, Scalar(255), FILLED);
    }

    // Apply morphological close
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

    // Find contours and pick the largest
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return Rect();
    }

    double maxArea = 0.0;
    int bestIdx = -1;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            bestIdx = i;
        }
    }
    if (maxArea < minArea) {
        return Rect();
    }

    return boundingRect(contours[bestIdx]);
}

int main(){
        VideoCapture cap(0);
        if(!cap.isOpened()){
                cout << "Error opening video stream or file" << endl;
                return -1;
        }
        string windowName = "object tracking";
        namedWindow(windowName, WINDOW_AUTOSIZE);
        
        Mat frame;

        while(true){
                cap >> frame;
                if(frame.empty()) break;
                
                // Detect chessboard using Shi-Tomasi corner detection
                Rect bbox = detectChessboardBBox(frame);
                
                // Draw bounding box if valid
                string status;
                if (bbox.width > 0 && bbox.height > 0) {
                    rectangle(frame, bbox, Scalar(0, 255, 0), 2);
                    status = "Chessboard Detected";
                    cout << "BoundingBox = (" << bbox.x << ", " << bbox.y << "), w=" 
                         << bbox.width << ", h=" << bbox.height << endl;
                } else {
                    status = "No Chessboard Detected";
                }
                
                // Draw status on frame
                putText(frame, status, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8, 
                       Scalar(0, 255, 0), 2);

                // Show result in the correctly named window
                imshow(windowName, frame);
                
                // Exit on ESC key
                char key = (char)waitKey(30);
                if(key == 27) break;
        }
        
        // Release resources
        cap.release();
        destroyAllWindows();
        
        return 0;
}
