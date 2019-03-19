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
        cv::vector<Point2f>& pt0,
        cv::vector<Point2f>& pt1,
        std::vector<size_t>& idx
        //std::vector<size_t>& i0,
        //std::vector<size_t>& i1,
        ){

    // TODO : do something about persistent kpt conversion
    cv::KeyPoint::convert(kf0.kpt_, pt0);
    pt1.reserve( pt0.size() );
    // cv::KeyPoint::convert(kf0.kpt_, pt1);

    std::vector<char> status; 
    std::vector<float> err;
    // TODO : support kf0.pyr_
    tracker_->calc(kf0.img_, kf1.img_,
            pt0, pt1,
            status, err);


    // TODO : support bidirectional check
}
