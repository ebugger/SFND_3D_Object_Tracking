
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2); //it2 is a ptr to Bbox
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1); // enclosingBoxes[0](it2)is the unique Bbox which the ptr(*it1) belongs to 
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox_prev, BoundingBox &boundingBox_curr, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double sum_distance = 0, mean_distance = 0;
    std::vector<cv::DMatch> match_in_roi;
    cout<<"Raw Matched size " << kptMatches.size()<<endl;

    double shrinkFactor = 0.05;
    cv::Rect smallerBox_prev, smallerBox_curr;

    smallerBox_prev.x = boundingBox_prev.roi.x + shrinkFactor * boundingBox_prev.roi.width / 2.0;
    smallerBox_prev.y = boundingBox_prev.roi.y + shrinkFactor * boundingBox_prev.roi.height / 2.0;
    smallerBox_prev.width = boundingBox_prev.roi.width * (1 - shrinkFactor);
    smallerBox_prev.height = boundingBox_prev.roi.height * (1 - shrinkFactor);

    smallerBox_curr.x = boundingBox_curr.roi.x + shrinkFactor * boundingBox_curr.roi.width / 2.0;
    smallerBox_curr.y = boundingBox_curr.roi.y + shrinkFactor * boundingBox_curr.roi.height / 2.0;
    smallerBox_curr.width = boundingBox_curr.roi.width * (1 - shrinkFactor);
    smallerBox_curr.height = boundingBox_curr.roi.height * (1 - shrinkFactor);

    //travesal the DMatches and find the kpt idx, and the use the KeyPoint to check if the bBox contains the kpt in both curr and prev.
    for(auto it = kptMatches.begin();it!=kptMatches.end();it++) {
        if(smallerBox_prev.contains(kptsCurr[it->trainIdx].pt) && smallerBox_curr.contains(kptsPrev[it->queryIdx].pt)) {
            //update the bBox kptMatches propertity
            match_in_roi.push_back(*it);
        }
    }
    cout<<"ROI Matched size " << match_in_roi.size()<<endl;

    //compute the mean distanve for all roi matched ptr
    for(auto it=match_in_roi.begin();it!=match_in_roi.end();it++) {
        double temp_dist;
        cv::KeyPoint prev_kpt = kptsPrev.at(it->queryIdx);
        cv::KeyPoint curr_kpt = kptsCurr.at(it->trainIdx);

        temp_dist = cv::norm(prev_kpt.pt - curr_kpt.pt);
        sum_distance += temp_dist;
    }
    mean_distance = sum_distance / match_in_roi.size();

    //cout<<"Mean distance "<< mean_distance<<endl;
    //filter the outlier by the threshold
    double ratio_dis_thresh = 1.7;
    for(auto it=match_in_roi.begin();it!=match_in_roi.end();it++) {
        double temp_dist;
        cv::KeyPoint prev_kpt = kptsPrev.at(it->queryIdx);
        cv::KeyPoint curr_kpt = kptsCurr.at(it->trainIdx);
        temp_dist = cv::norm(prev_kpt.pt - curr_kpt.pt);
        //cout<<"current distance "<< temp_dist<<endl;
        if(temp_dist < ratio_dis_thresh * mean_distance) {
            boundingBox_curr.kptMatches.push_back(*it);
        }
    }
    cout<<"Ratio Matched size " << boundingBox_curr.kptMatches.size()<<endl;   

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    //int n_element = sizeof(distRatios) / sizeof(double);
    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0); //floor(2.6)==2 ceil(2.6)==3 round(2.6)==3
    double medianDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex-1] + distRatios[medIndex]) / 2 : distRatios[medIndex];
    double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);
    TTC = -dT / (1 - medianDistRatio - std::numeric_limits<double>::epsilon());
    
    cout<<"Cemera distance Ratio from Current bBox: " << medianDistRatio <<" of size: " << distRatios.size()<<endl;
    cout<<"TTC based on Cemera is: " << TTC << endl;

    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
    long medIndex; // ...
    double median_Prev_x, median_Curr_x;
    
    cv::Point mean_Prev, mean_Curr;
    //double temp_cx, temp_cy, temp_px, temp_py;
    std::vector<double> temp_px, temp_cx;
    for(auto it=lidarPointsPrev.begin();it!=lidarPointsPrev.end();it++) {
            temp_px.push_back(it->x);       
    }
    std::sort(temp_px.begin(), temp_px.end());
    medIndex = floor(temp_px.size() / 2.0);
    median_Prev_x = temp_px.size() % 2 ? (temp_px[medIndex-1] + temp_px[medIndex]) / 2 : temp_px[medIndex];

    //TTC = -dT / (1 - meanDistRatio);
    for(auto it=lidarPointsCurr.begin();it!=lidarPointsCurr.end();it++) {
            temp_cx.push_back(it->x);    

    }

    if(temp_px.size() && temp_cx.size()) {
        std::sort(temp_cx.begin(), temp_cx.end());
        medIndex = floor(temp_cx.size() / 2.0);
        median_Curr_x = temp_cx.size() % 2 ? (temp_cx[medIndex-1] + temp_cx[medIndex]) / 2 : temp_cx[medIndex];

        TTC = median_Curr_x * dT / (median_Prev_x - median_Curr_x);
        cout<<"Lidar ptr Median x from previous bBox: " << median_Prev_x <<endl;
        cout<<"Lidar ptr Median x from Current bBox: " << median_Curr_x <<endl;
        cout<<"TTC based on Lidar is: " << TTC << endl;
    }else {
        TTC = NAN;
        return;
    }


}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    
    int l_prevBbox = prevFrame.boundingBoxes.size();
    int l_currBbox = currFrame.boundingBoxes.size();
    //2D array to store all the pair counts
    int ab[l_prevBbox][l_currBbox] = { }; 

    //std::vector<int> query_id, train_id;
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        //KeyPoint coordinate
        cv::Point query = prevFrame.keypoints[it->queryIdx].pt;
        cv::Point train = currFrame.keypoints[it->trainIdx].pt;

        //store the bboxes IDs for which each point belongs to, one ptr may result in multiple bboxes.
        std::vector<int> query_id, train_id; //inside the iterator!

        //extract the bbox IDs which contains the keypoints for the previous frame
        for(auto it2=prevFrame.boundingBoxes.begin();it2!=prevFrame.boundingBoxes.end();++it2) {
            if(it2->roi.contains(query)) {
                query_id.push_back(it2->boxID);
            }
        }

        //extract the bbox IDs which contains the keypoints for the currents frame
        for(auto it2=currFrame.boundingBoxes.begin();it2!=currFrame.boundingBoxes.end();++it2) {
            if(it2->roi.contains(train)) {
                train_id.push_back(it2->boxID);
            }
        }      

        //update the counts for the possiable combination pair in 2D array
        if(query_id.size() && train_id.size()) {
            for(int x : query_id) {
                for(int y : train_id) {
                    ab[x][y] += 1;
                }
            }            
        }
    }

    //traverse the 2D array by row(from previous frame), update the maxium value and extract the index for the paired(current frame) bbox ID.
    for(int i=0;i<l_prevBbox;i++) {
        int curr_idx = 0;
        int max_c = 0;
        for(int j=0;j<l_currBbox;j++) {
            if(ab[i][j] > max_c) {
                max_c = ab[i][j]; 
                curr_idx = j;
            }
        }
        //if(max_c>0){
            //map(i,j) === (prev, curr)
            bbBestMatches[i] = curr_idx;
            cout<<"Previous bbox#: " << i << " matched --> Current bbox#： "<< bbBestMatches[i] << endl;
        //}

    }

    // ...
}
