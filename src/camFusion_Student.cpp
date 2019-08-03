
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
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    
    //store the bboxes IDs for which each point belongs to, one ptr may result in multiple bboxes.
    //std::vector<int> pre_result, curr_result;

    int p_IDs = prevFrame.boundingBoxes.size();
    int c_IDs = currFrame.boundingBoxes.size();
    //2D array to store all the pair counts
    int ab[p_IDs][c_IDs] = {}; 
    //bool pb=false, cb = false;
    std::vector<int> query_id, train_id;
    for (auto it = matches.begin(); it != matches.end() - 1; ++it) {
        //extract the bbox IDs which contains the keypoints for the previous frame
        //for(auto it2=prevFrame.boundingBoxes.begin();it2!=prevFrame.boundingBoxes.end();++it2) {
            //int idx_kpt = it1->queryIdx;
            //cv::Point &kpt = prevFrame.keypoints[it1->queryIdx].pt;
        //    if(it2->roi.contains(prevFrame.keypoints[it1->queryIdx].pt)) {
        //        pre_result.push_back(it2->boxID);
        //    }
        //}
        std::vector<int> pre_result, curr_result;
        //cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
       // auto query_pt = cv::Point(query.pt.x, query.pt.y);
        //auto query_pt = query.pt;
        //cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        //auto train_pt = train.pt;

        //cv::Point prev_p = cv::Point(prevFrame.keypoints[it1->queryIdx].pt.x, prevFrame.keypoints[it1->queryIdx].pt.y);
        for(auto it2=prevFrame.boundingBoxes.begin();it2!=prevFrame.boundingBoxes.end();++it2) {
        //for(int i=0;i<p_IDs;i++) {
            //if(prevFrame.boundingBoxes[i].roi.contains(query_pt)) {
            if(it2->roi.contains(prevFrame.keypoints[it->queryIdx].pt)) {
                //pb = true;
                pre_result.push_back(it2->boxID);
            }
    }

        //extract the bbox IDs which contains the keypoints for the currents frame
        //for(int i=0;i<c_IDs;i++) {
        for(auto it3=prevFrame.boundingBoxes.begin();it3!=prevFrame.boundingBoxes.end();++it3) {
            if(it3->roi.contains(currFrame.keypoints[it->trainIdx].pt)) {
                //cb = true;
                curr_result.push_back(it3->boxID);
            }
        }
      
/*

    for (auto it = matches.begin(); it != matches.end() - 1; ++it)     {
        cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query.pt.x, query.pt.y);
        bool pb = false;
        cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train.pt.x, train.pt.y);
        bool cb = false;
        std::vector<int> query_id, train_id;
*/
        //if(pre_result.size()>0 && curr_result.size()>0) {
        if(pre_result.size() && curr_result.size()) {
            for(int x : pre_result) {
                for(int y : curr_result) {
                    ab[x][y] += 1;
                }
            }            
        }


    }

    //traverse the 2D array by row(from previous frame), update the maxium value and extract the index for the paired(current frame) bbox ID.
    for(int i=0;i<p_IDs;i++) {
        int curr_idx = 0;
        int max_c = 0;
        for(int j=0;j<c_IDs;j++) {
            if(ab[i][j] > max_c) {
                max_c = ab[i][j]; 
                curr_idx = j;
            }
        }
        if(max_c>0) {
            bbBestMatches[i] = curr_idx;
            cout<<"Previous bbox#: " << i << " matched --> Current bbox#： "<< bbBestMatches[i] << endl;
        }

    }

    // ...
}

