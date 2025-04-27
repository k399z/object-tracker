#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Rect detect_chessboard_bbox(const Mat& frame, int min_area = 5000) {
    Mat gray, thresh; // Removed unused blurred, dst, dst_dilated

    // 灰度 + 高斯模糊
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // --- Shi-Tomasi Corner Detection ---
    vector<Point2f> corners;
    // Parameters: image, corners, maxCorners, qualityLevel, minDistance
    goodFeaturesToTrack(gray, corners, 100, 0.01, 10);

    if (corners.empty() || corners.size() < 4) { // Need at least 4 corners for a reasonable shape
        return Rect();
    }

    // Create a mask with detected corners
    thresh = Mat::zeros(gray.size(), CV_8U); // Initialize thresh here
    for (const auto &corner : corners) {
        // Use integer coordinates for drawing on the mask
        circle(thresh, Point(cvRound(corner.x), cvRound(corner.y)), 5, Scalar(255), FILLED);
    }
    // --- End Shi-Tomasi ---

    // 闭操作 (Morphological Close) - Keep this part
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15)); // Adjusted kernel size slightly
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

    // 查找轮廓 (Find Contours) - Keep this part
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return Rect();

    // 找最大轮廓 (Find Largest Contour) - Keep this part
    double max_area = 0;
    int best_index = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            best_index = i;
        }
    }

    if (best_index == -1 || max_area < min_area)
        return Rect();

    return boundingRect(contours[best_index]);
}

int main() {
    VideoCapture cap(0);
            cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        double t0 = getTickCount();  // ⏱ FPS开始

        Rect bbox = detect_chessboard_bbox(frame);
        if (bbox.area() > 0) {
            rectangle(frame, bbox, Scalar(0, 255, 0), 2);
            cout << "BoundingBox = (" << bbox.x << ", " << bbox.y 
                 << "), w=" << bbox.width << ", h=" << bbox.height << endl;
        }

        // ⏱ FPS统计
        double t1 = getTickCount();
        double fps = getTickFrequency() / (t1 - t0);
        putText(frame, format("FPS: %.2f", fps), Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Chessboard Detection", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
