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

// Previous code used CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY every frame (slow).
// Split into fast (no extra flags) and accurate (refinement only for small ROI).
static const int CB_FLAGS_FAST     = 0;
static const int CB_FLAGS_ACCURATE = CALIB_CB_ACCURACY;
// Tracking robustness additions
static int consecutiveROIMisses = 0;
static const int ROI_MISS_RESET = 5;           // After this many misses, discard last bbox.
// Add grace & smoothing
static const int GRACE_FRAMES = 6;             // Keep showing last bbox this many miss frames.
static int missGrace = 0;
static const float SMOOTH_ALPHA = 0.30f; // base (still used as fallback)
static float adaptiveAlpha = SMOOTH_ALPHA;
static const float MIN_IOU_ACCEPT = 0.15f; // reject wild outliers (except first)
static Rect2f smooth_bbox;    // <--- re-added
static bool hasSmooth = false; // <--- re-added
// Pre-allocate Mats and vectors
static Mat gray_full, small;
static vector<Point2f> corners;

// Timing + tracking state
static Rect last_bbox_small;
static double lastDetectMs = 0.0;
static int frameCounter = 0;
static const int FULL_DETECT_INTERVAL = 3; // Only try full image every N frames when lost.

// Modified: add expandFrac parameter (default 0.10f)
static bool verifyChessboardInROI(const Mat& gray, const Rect& bbox, Rect& chessRect,
                                  vector<Point2f>& chessCorners, bool accurate, float expandFrac = 0.10f) {
    if (bbox.width <= 0 || bbox.height <= 0) return false;
    Rect roi = bbox & Rect(0, 0, gray.cols, gray.rows);
    if (roi.width <= 0 || roi.height <= 0) return false;

    // Expand ROI a bit to avoid tight cropping
    int dx = std::max(2, (int)(roi.width  * expandFrac));
    int dy = std::max(2, (int)(roi.height * expandFrac));
    roi.x = std::max(0, roi.x - dx);
    roi.y = std::max(0, roi.y - dy);
    roi.width  = std::min(gray.cols  - roi.x, roi.width  + 2 * dx);
    roi.height = std::min(gray.rows - roi.y, roi.height + 2 * dy);

    // Use local gray_roi instead of static
    Mat gray_roi = gray(roi);
    int flags = accurate ? CB_FLAGS_ACCURATE : CB_FLAGS_FAST;
    // no equalizeHist(gray_roi, gray_roi);

    // FAST_CHECK will abort quickly on bad regions
    bool found = findChessboardCornersSB(gray_roi, PATTERN, corners, flags);
    if (!found || corners.size() != PATTERN.area())
        return false;

    Rect localRect = boundingRect(corners);
    chessRect = Rect(localRect.tl()+roi.tl(), localRect.br()+roi.tl());
    chessCorners = corners;
    for (auto& p : chessCorners) p += Point2f((float)roi.x,(float)roi.y);
    return true;
}

// Wrapper Function to detect chessboard bounding box
// Accept grayscale image directly, avoid redundant conversion
Rect detectChessboardBBox(const Mat& gray) {
    // Try robust full-frame chessboard detection (most reliable).
    Rect full(0, 0, gray.cols, gray.rows);
    Rect chessRect;
    vector<Point2f> chessCorners;
    if (verifyChessboardInROI(gray, full, chessRect, chessCorners, false, 0.02f))
        return chessRect;
    else
        return Rect();
}

// New: ROI-first attempt (accurate on small area)
static Rect detectChessboardBBoxROI(const Mat& gray, const Rect& prev) {
    if (prev.width > 0 && prev.height > 0) {
        Rect chessRect;
        vector<Point2f> chessCorners;
        if (verifyChessboardInROI(gray, prev, chessRect, chessCorners, true, 0.30f))
            return chessRect;
    }
    return Rect();
}

// (Optional) small helper
static float rectIoU(const Rect2f& a, const Rect2f& b) {
    Rect2f inter = a & b;
    float ia = inter.area();
    if (ia <= 0) return 0.f;
    float ua = a.area() + b.area() - ia;
    return ua > 0 ? ia / ua : 0.f;
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

                // Apply GaussianBlur once here
                GaussianBlur(small, small, Size(5, 5), 0);

                frameCounter++;

                // Always permit periodic full-frame attempts (independent of last bbox)
                bool allowFull = (frameCounter % FULL_DETECT_INTERVAL == 0);

                int64 t0 = getTickCount();
                Rect bbox_small;
                bool roiTried = false;

                // 1) ROI attempt if we have a previous bbox
                if (last_bbox_small.area() > 0) {
                    roiTried = true;
                    bbox_small = detectChessboardBBoxROI(small, last_bbox_small);
                }

                // 2) If ROI failed, optionally do full-frame (fast flags)
                if (bbox_small.area() == 0) {
                    if (roiTried) consecutiveROIMisses++;
                    if (allowFull) {
                        Rect fullCandidate = detectChessboardBBox(small);
                        if (fullCandidate.area() > 0) {
                            bbox_small = fullCandidate;
                            consecutiveROIMisses = 0;
                        }
                    }
                    // If too many consecutive ROI misses without recovery, drop tracking
                    if (consecutiveROIMisses >= ROI_MISS_RESET) {
                        last_bbox_small = Rect();
                        consecutiveROIMisses = 0; // reset counter after giving up
                    }
                }

                // 3) On success update last bbox and smoothing (always update AND gate by IoU)
                if (bbox_small.area() > 0) {
                    Rect2f det((float)bbox_small.x,(float)bbox_small.y,
                               (float)bbox_small.width,(float)bbox_small.height);

                    bool accept = true;
                    if (hasSmooth) {
                        float iou = rectIoU(det, smooth_bbox);
                        if (iou < MIN_IOU_ACCEPT) {
                            // Treat as suspicious outlier: only accept if we already missed several times
                            accept = (missGrace > 2);
                        }
                    }

                    if (accept) {
                        last_bbox_small = bbox_small;
                        consecutiveROIMisses = 0;
                        missGrace = 0;

                        if (!hasSmooth) {
                            smooth_bbox = det;
                            hasSmooth = true;
                        } else {
                            // Adaptive alpha based on normalized center shift
                            Point2f cOld(smooth_bbox.x + smooth_bbox.width *0.5f,
                                         smooth_bbox.y + smooth_bbox.height*0.5f);
                            Point2f cNew(det.x + det.width*0.5f,
                                         det.y + det.height*0.5f);
                            float maxDim = std::max(smooth_bbox.width, smooth_bbox.height);
                            float shiftNorm = maxDim > 1.f ? (float)norm(cNew - cOld) / maxDim : 0.f;
                            if      (shiftNorm > 0.40f) adaptiveAlpha = 0.70f;
                            else if (shiftNorm > 0.25f) adaptiveAlpha = 0.55f;
                            else if (shiftNorm > 0.12f) adaptiveAlpha = 0.40f;
                            else                        adaptiveAlpha = 0.20f;

                            smooth_bbox.x      = adaptiveAlpha * det.x      + (1.f - adaptiveAlpha) * smooth_bbox.x;
                            smooth_bbox.y      = adaptiveAlpha * det.y      + (1.f - adaptiveAlpha) * smooth_bbox.y;
                            smooth_bbox.width  = adaptiveAlpha * det.width  + (1.f - adaptiveAlpha) * smooth_bbox.width;
                            smooth_bbox.height = adaptiveAlpha * det.height + (1.f - adaptiveAlpha) * smooth_bbox.height;
                        }
                    } else {
                        // Outlier not accepted -> treat like a miss (do not reset grace fully)
                        missGrace++;
                    }
                } else {
                    missGrace++;
                    if (missGrace > ROI_MISS_RESET + GRACE_FRAMES) {
                        last_bbox_small = Rect();
                        hasSmooth = false;
                    }
                }

                lastDetectMs = (getTickCount() - t0) * 1000.0 / getTickFrequency();

                // Unified display: only ever draw smoothed bbox (no raw bounce)
                Rect bbox;
                if (hasSmooth && missGrace <= GRACE_FRAMES) {
                    double fx = (double)frame.cols / small.cols;
                    double fy = (double)frame.rows / small.rows;
                    Rect2f sb = smooth_bbox;
                    bbox = Rect(
                        cvRound(sb.x * fx), cvRound(sb.y * fy),
                        cvRound(sb.width * fx), cvRound(sb.height * fy)
                    );
                }

                // Status + draw
                string status;
                if (bbox.area() > 0) {
                    rectangle(frame, bbox, Scalar(0,255,0), 2);
                    status = missGrace == 0 ? "Chessboard (stable)" : "Chessboard (hold)";
                    cout << "BBox (" << bbox.x << "," << bbox.y << ") w=" << bbox.width
                         << " h=" << bbox.height
                         << " detect=" << lastDetectMs << "ms"
                         << " missGrace=" << missGrace
                         << " alpha=" << adaptiveAlpha
                         << endl;
                } else {
                    status = (frameCounter % FULL_DETECT_INTERVAL == 0) ? "Searching (full scan)" : "Lost";
                }

                // Draw status on downscaled frame
                putText(frame, status, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                       Scalar(0, 255, 0), 2);

                // Draw FPS (previous frame's throughput)
                putText(frame, cv::format("FPS: %.1f", fps), Point(30, 60), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 0), 2);
                putText(frame, cv::format("Detect: %.1f ms", lastDetectMs), Point(30, 90),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);

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
