#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

// 1) At top add constants and pre-allocate temps
static const Size PROC_SIZE(320, 240);
static const Size PATTERN(11, 8);
// Drop CALIB_CB_FAST_CHECK – SB version doesn’t support it
static const int  CB_FLAGS = CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY;

// Pre-allocate Mats and vectors
static Mat gray_full, gray_roi, small;
static vector<Point2f> corners;

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

    gray_roi = gray(roi);                        // reuse gray_roi
    // no equalizeHist(gray_roi, gray_roi);

    // FAST_CHECK will abort quickly on bad regions
    bool found = findChessboardCornersSB(
        gray_roi, PATTERN, corners, CB_FLAGS
    );
    if (!found || corners.size() != PATTERN.area())
        return false;

    // optionally skip subpix for speed
    // cornerSubPix(gray_roi, corners, Size(11,11), Size(-1,-1),
    //              TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,30,0.1));

    Rect localRect = boundingRect(corners);
    chessRect = Rect(localRect.tl()+roi.tl(), localRect.br()+roi.tl());
    chessCorners = corners;
    for (auto& p : chessCorners) p += Point2f((float)roi.x,(float)roi.y);
    return true;
}

// Function to detect chessboard bounding box
Rect detectChessboardBBox(Mat& frame, int minArea = 5000) {
    Mat gray;
    // Replace unconditional cvtColor with channel check:
    if (frame.channels() == 3) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
    } else {
        // already gray
        gray = frame;
    }
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // Try robust full-frame chessboard detection (most reliable).
    Rect full(0, 0, gray.cols, gray.rows);
    Rect chessRect;
    vector<Point2f> chessCorners;
    if (verifyChessboardInROI(gray, full, chessRect, chessCorners)) {
        for (const auto& p : chessCorners) {
            circle(frame, p, 3, Scalar(0, 0, 255), FILLED);
        }
        return chessRect;
    }

    // No fallback: return empty if not found.
    return Rect();
}

int main(){
        VideoCapture cap(0);
        if(!cap.isOpened()){
                cout << "Error opening video stream or file" << endl;
                return -1;
        }
        // Hint camera to capture at 640x480
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        string windowName = "object tracking";
        namedWindow(windowName, WINDOW_AUTOSIZE);

        // FPS state
        int64 lastTick = cv::getTickCount();
        double fps = 0.0;

        // Pre-allocate images once
        Mat frame;

        while(true){
                // Update FPS from previous iteration
                int64 nowTick = cv::getTickCount();
                double dt = (nowTick - lastTick) / cv::getTickFrequency();
                if (dt > 0) fps = 1.0 / dt;
                lastTick = nowTick;

                cap >> frame;
                if(frame.empty()) break;

                // single BGR→GRAY conversion
                cvtColor(frame, gray_full, COLOR_BGR2GRAY);

                // downscale aggressively
                resize(gray_full, small, PROC_SIZE, 0,0, INTER_AREA);

                // detect on small
                Rect bbox_small = detectChessboardBBox(small);
                // scale bbox back to display size
                Rect bbox(
                    bbox_small.x*2, bbox_small.y*2,
                    bbox_small.width*2, bbox_small.height*2
                );

                // Draw bounding box if valid (on downscaled frame)
                string status;
                if (bbox.width > 0 && bbox.height > 0) {
                    rectangle(frame, bbox, Scalar(0, 255, 0), 2);
                    status = "Chessboard Grid Verified";
                    cout << "BoundingBox = (" << bbox.x << ", " << bbox.y << "), w="
                         << bbox.width << ", h=" << bbox.height << endl;
                } else {
                    status = "No Chessboard Grid";
                }

                // Draw status on downscaled frame
                putText(frame, status, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                       Scalar(0, 255, 0), 2);

                // Draw FPS (previous frame's throughput)
                putText(frame, cv::format("FPS: %.1f", fps), Point(30, 60), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 0), 2);

                // Show the downscaled frame
                imshow(windowName, frame);

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
