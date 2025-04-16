#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

// Function to detect chessboard bounding box using only Canny edge detection
Rect detectChessboardBBox(Mat& frame, int minArea = 5000) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    
    // Enhanced edge detection with better parameters
    Mat edges;
    Canny(gray, edges, 30, 150);
    
    // Apply morphological operations to strengthen edges
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(edges, edges, kernel);
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);
    
    // Find contours
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return Rect();
    }
    
    // Filter contours by area and shape
    vector<vector<Point>> validContours;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < minArea) {
            continue;
        }
        
        // Check if shape is approximately rectangular
        double peri = arcLength(contour, true);
        vector<Point> approx;
        approxPolyDP(contour, approx, 0.04 * peri, true);
        
        // Check if it has at least 4 corners (rectangular) and is convex
        if (approx.size() >= 4 && isContourConvex(approx)) {
            // Check if aspect ratio is reasonable for a chessboard (not too elongated)
            Rect boundRect = boundingRect(contour);
            float aspectRatio = (float)boundRect.width / boundRect.height;
            if (0.7 <= aspectRatio && aspectRatio <= 1.3) {  // Close to square
                validContours.push_back(contour);
            }
        }
    }
    
    if (validContours.empty()) {
        return Rect();
    }
    
    // Choose the largest valid contour
    int bestIdx = 0;
    double largestArea = 0;
    for (int i = 0; i < validContours.size(); i++) {
        double area = contourArea(validContours[i]);
        if (area > largestArea) {
            largestArea = area;
            bestIdx = i;
        }
    }
    
    return boundingRect(validContours[bestIdx]);
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
                
                // Detect chessboard using Canny edge detection
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
