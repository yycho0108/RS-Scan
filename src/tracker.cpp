#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include "tracker.hpp"
#include <iostream>

Tracker::Tracker(){
    tracker_ = cv::SparsePyrLKOpticalFlow::create(
            //cv::Size(21,21), //winSize,
            //3, //maxLevel,
            //criteria,
            //flags,
            //minEigThreshold
            );
}

Tracker::~Tracker(){

}

void Tracker::track(
        const FrameData& kf0,
        const FrameData& kf1,
        std::vector<cv::Point2f>& pt0,
        std::vector<cv::Point2f>& pt1,
        std::vector<cv::DMatch>& m01
        ){

    if( pt0.size() <= 0) {
        // pt0 == kf0.kpt_
        cv::KeyPoint::convert(kf0.kpt_, pt0);
        pt1.clear();
        pt1.reserve( pt0.size() );
        // cv::KeyPoint::convert(kf0.kpt_, pt1);
        // TODO : do something about persistent kpt conversion
    }

    std::vector<uint8_t> status; 
    std::vector<float> err;
    // TODO : support kf0.pyr_
    tracker_->calc(kf0.gray_, kf1.gray_,
            pt0, pt1,
            status, err);

    // TODO : support bidirectional check
    m01.clear();
    m01.reserve( pt0.size() );
    for(int i=0; i<pt0.size(); ++i){
        if(!status[i]) continue;
        //queryIdx, trainIdx, distance
        m01.emplace_back(i, i, err[i]);
    }
}
