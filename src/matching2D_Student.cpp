
#include <numeric>
#include "matching2D.hpp"

using namespace std;
/* */
// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        //int normType = cv::NORM_HAMMING;
        //matcher = cv::BFMatcher::create(normType, crossCheck);
        //int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        //int normType = descSource.type() != CV_32F ? cv::NORM_HAMMING : cv::NORM_L2;
        int normType = descSource.type() == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        //int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(/* descSource.type() != CV_32F ||*/ descRef.type() != CV_32F ) {
            descSource.convertTo(descSource, CV_32F);
            //cout<<"descSource/prev type: " << descSource.type()<<endl;
            descRef.convertTo(descRef, CV_32F);
            //cout<<"descSource/curr: " << descRef.type()<<endl;
        }
        cout<<"descSource.type: " << descSource.type()<<endl;
        cout<<"descRef.type: " << descRef.type()<<endl;
        //cout<<"descSource.empty: " << descSource.empty()<<endl;
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> matches_knn;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, matches_knn, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << " (KNN) with n=" << matches_knn.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        // TODO : filter matches using descriptor distance ratio test
        const float rationThreshold = 0.8f;
        for(auto it=matches_knn.begin();it!=matches_knn.end();it++) {
            if((*it)[0].distance < rationThreshold * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }    

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        //...
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
        //...
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        //extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        extractor = cv::xfeatures2d::SIFT::create();
        //...
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
        //...
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    if (descriptorType.compare("AKAZE") == 0)
    extractor->detectAndCompute(img, cv::noArray(), keypoints, descriptors,false);
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {
    string windowname; 
    if(detectorType.compare("HARRIS") == 0) {
        int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
        int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
        int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix, float value
        double k = 0.04;       // Harris parameter (see equation for details)

        // Detect Harris corners and normalize output
        double t = (double)cv::getTickCount();
        cv::Mat dst, dst_norm, dst_norm_scaled;
        dst = cv::Mat::zeros(img.size(), CV_32FC1);
        cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);    

        //NMS
        double maxOverlap = 0.0;

        for(size_t i=0;i<dst_norm.rows;i++){
            for(size_t j=0;j<dst_norm.cols;j++) {
                int response = (int)dst_norm.at<float>(i,j); //Harris response matrix, its a float matrix
                if(response > minResponse) { //store the pointd above the threshold
                    cv::KeyPoint kpt;
                    kpt.pt = cv::Point2f(j,i); //point coords, you shoule switch the i, j!!!
                    kpt.size = 2 * apertureSize; //size
                    kpt.response = response; //response value

                    bool bOverlap = false;
                    for(auto it = keypoints.begin(); it!=keypoints.end(); ++it) {
                        double kptOverldap = cv::KeyPoint::overlap(kpt, *it);
                        if(kptOverldap > maxOverlap) {
                            bOverlap = true; // first to determine overlap, then compare response value
                            if(kpt.response > it->response) {
                                *it = kpt;
                                break;  //successfuly replace/update one KeyPoint and exit, as others left are already have processed not overlap
                            }
                        }
                    }
                    if(!bOverlap)
                        keypoints.push_back(kpt); //directly store the ptr not below the threshold and ready to be compred in next response map loop
                }
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "Detector: Harris with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        img = dst_norm_scaled;
        windowname = "Harris Corner Detector Result";       
    }
    if(detectorType.compare("FAST") == 0) {
        int fastThreshold = 30;
        bool fastNMS = true;
        cv::FastFeatureDetector::DetectorType fastType = cv::FastFeatureDetector::TYPE_9_16;

        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(fastThreshold, fastNMS, fastType);

        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        std::cout<< "Detector: Fast with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl; 
        windowname = "FAST Detector Result";   
             
    }
    if(detectorType.compare("BRISK") == 0) {

        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "Detector: BRISK with n= " << endl;
        //total_time += 1000 * t / 1.0;
        windowname = "BRISK Detector Result";   

        /*
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
        cv::Mat descBRISK;
        t = (double)cv::getTickCount();
        descriptor->compute(imgGray, keypoints, descBRISK);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;
        */
        // visualize results
      
    }
    if(detectorType.compare("SIFT") == 0) {

        cv::Ptr<cv::FeatureDetector> detector_sift = cv::xfeatures2d::SIFT::create();

        double t = (double)cv::getTickCount();
        detector_sift->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t ) / cv::getTickFrequency();
        std::cout << "Detector: SIFT with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowname = "SIFT Detector Result"; 

    }
    if(detectorType.compare("ORB") == 0) {
        cv::Ptr<cv::FeatureDetector> detector_orb = cv::ORB::create();

        double t = (double)cv::getTickCount();
        detector_orb->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        std::cout<< "detector: ORB with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowname = "ORB Detector Result";
    
    }
    if(detectorType.compare("AKAZE") == 0) {
        //cv::AKAZE::DescriptorType a_type = cv::AKAZE::DescriptorType::DESCRIPTOR_MLDB;
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        std::cout<< "detector: AKAZE with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl; 
        windowname = "AKAZE Detector Result";

    } 

    if(bVis) {
        cv::Mat visImg = img.clone();
        cv::drawKeypoints(img, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowname, 7);
        cv::imshow(windowname, visImg);
        cv::waitKey(0);        
    }    
}