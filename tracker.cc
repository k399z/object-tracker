#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

// Helper: verify chessboard grid inside ROI using common inner-corner sizes
static bool verifyChessboardInROI(const Mat& gray, const Rect& bbox, Rect& chessRect, vector<Point2f>& chessCorners) {
    if (bbox.width <= 0 || bbox.height <= 0) return false;
    Rect roi = bbox & Rect(0, 0, gray.cols, gray.rows);
    if (roi.width <= 0 || roi.height <= 0) return false;

    // Expand ROI a bit to avoid tight cropping
    int dx = std::max(2, (int)(roi.width * 0.10f));
    int dy = std::max(2, (int)(roi.height * 0.10f));
    roi.x = std::max(0, roi.x - dx);
    roi.y = std::max(0, roi.y - dy);
    roi.width  = std::min(gray.cols  - roi.x, roi.width  + 2 * dx);
    roi.height = std::min(gray.rows - roi.y, roi.height + 2 * dy);

    Mat grayRoi = gray(roi);
    if (grayRoi.empty()) return false;
    if (grayRoi.channels() != 1) cvtColor(grayRoi, grayRoi, COLOR_BGR2GRAY);

    // Improve contrast for detection
    equalizeHist(grayRoi, grayRoi);

    // Common inner-corner patterns (cols, rows)
    vector<Size> patterns = {
        Size(9,6), Size(7,7), Size(8,6), Size(8,5), Size(7,5), Size(6,5), Size(5,4), Size(4,3)
    };

    for (const auto& pattern : patterns) {
        vector<Point2f> corners;

        bool found = findChessboardCornersSB(
            grayRoi, pattern, corners, CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY
        );
        if (!found) continue;

        if (corners.size() != static_cast<size_t>(pattern.width * pattern.height)) continue;

        cornerSubPix(
            grayRoi, corners, Size(11,11), Size(-1,-1),
            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1)
        );

        Rect localRect = boundingRect(corners);
        chessRect = Rect(localRect.tl() + roi.tl(), localRect.br() + roi.tl());
        chessCorners = corners;
        for (auto& p : chessCorners) p += Point2f((float)roi.x, (float)roi.y);
        return true;
    }
    return false;
}

// Function to detect chessboard bounding box using Shi-Tomasi corner detection
// It now also draws the detected corners onto the input frame for debugging.
Rect detectChessboardBBox(Mat& frame, int minArea = 5000) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // First, try robust full-frame chessboard detection (most reliable).
    {
        Rect full(0, 0, gray.cols, gray.rows);
        Rect chessRect;
        vector<Point2f> chessCorners;
        if (verifyChessboardInROI(gray, full, chessRect, chessCorners)) {
            // Draw verified chessboard corners (distinct color)
            for (const auto& p : chessCorners) {
                circle(frame, p, 3, Scalar(0, 0, 255), FILLED);
            }
            return chessRect;
        }
    }

    // Fallback: Shi–Tomasi corner clustering
    vector<Point2f> corners;
    goodFeaturesToTrack(gray, corners, 100, 0.01, 10);

    // --- Debug: Draw detected corners ---
    RNG rng(12345); // Random number generator for colors
    for (size_t i = 0; i < corners.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)); // Random color
        circle(frame, corners[i], 4, color, FILLED); // Draw small filled circles
    }
    // --- End Debug ---

    if (corners.empty() || corners.size() < 4) {
        return Rect();
    }

    // Create a mask with detected corners
    Mat thresh = Mat::zeros(gray.size(), CV_8UC1);
    for (auto &corner : corners) {
        circle(thresh, Point(cvRound(corner.x), cvRound(corner.y)), 5, Scalar(255), FILLED);
    }

    // Apply morphological close (scale kernel with image size)
    int k = std::max(7, (int)(0.02 * std::min(gray.cols, gray.rows)));
    if ((k % 2) == 0) ++k;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

    // Find contours and pick the largest
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return Rect();

    double maxArea = 0.0;
    int bestIdx = -1;
    for (int i = 0; i < (int)contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            bestIdx = i;
        }
    }

    // Scale min area with image size; keep user-provided floor
    int adaptiveMinArea = std::max(minArea, (int)(gray.total() * 0.002)); // ~0.2% of image
    if (maxArea < adaptiveMinArea || bestIdx == -1) {
        return Rect();
    }

    // Verify the candidate box contains a chessboard grid
    Rect candidate = boundingRect(contours[bestIdx]);
    Rect chessRect;
    vector<Point2f> chessCorners;
    if (!verifyChessboardInROI(gray, candidate, chessRect, chessCorners)) {
        return Rect();
    }

    // Optional: draw verified chessboard corners (distinct color)
    for (const auto& p : chessCorners) {
        circle(frame, p, 3, Scalar(0, 0, 255), FILLED);
    }

    return chessRect;
}

int main(){
        VideoCapture cap(0);
        if(!cap.isOpened()){
                cout << "Error opening video stream or file" << endl;
                return -1;
        }
        // Hint camera to capture at width 640 (driver may adjust height automatically)
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        string windowName = "object tracking";
        namedWindow(windowName, WINDOW_AUTOSIZE);

        // FPS state
        int64 lastTick = cv::getTickCount();
        double fps = 0.0;

        Mat frame;

        while(true){
                // Update FPS from previous iteration
                int64 nowTick = cv::getTickCount();
                double dt = (nowTick - lastTick) / cv::getTickFrequency();
                if (dt > 0) fps = 1.0 / dt;
                lastTick = nowTick;

                cap >> frame;
                if(frame.empty()) break;

                // Downscale to max width 640 for faster processing
                Mat proc;
                if (frame.cols > 640) {
                        float scale = 640.0f / frame.cols;
                        resize(frame, proc, Size(), scale, scale, INTER_AREA);
                } else {
                        proc = frame; // no resize
                }

                // Detect chessboard on the (possibly) downscaled frame
                Rect bbox = detectChessboardBBox(proc);

                // Draw bounding box if valid (on downscaled frame)
                string status;
                if (bbox.width > 0 && bbox.height > 0) {
                    rectangle(proc, bbox, Scalar(0, 255, 0), 2);
                    status = "Chessboard Grid Verified";
                    cout << "BoundingBox = (" << bbox.x << ", " << bbox.y << "), w="
                         << bbox.width << ", h=" << bbox.height << endl;
                } else {
                    status = "No Chessboard Grid";
                }

                // Draw status on downscaled frame
                putText(proc, status, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                       Scalar(0, 255, 0), 2);

                // Draw FPS (previous frame's throughput)
                putText(proc, cv::format("FPS: %.1f", fps), Point(30, 60), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 0), 2);

                // Show the downscaled frame
                imshow(windowName, proc);

                // Exit on ESC or 'q'
                char key = (char)waitKey(30);
                if(key == 'q') break;
                if(key == 27) break;
        }

        // Release resources
        cap.release();
        destroyAllWindows();

        return 0;
}
