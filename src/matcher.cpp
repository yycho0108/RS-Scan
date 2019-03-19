#include "matcher.hpp"
#include <iostream>

Matcher::Matcher(const cv::Mat& K, float lowe, float maxd):
    K_(K), lowe_(lowe), maxd_(maxd),
    matcher_(cv::makePtr<cv::flann::LshIndexParams>(12,20,2)){
        //TODO : tune matcher params?
    }

void Matcher::filter(
        const std::vector<std::vector<cv::DMatch>>& m_knn,
        std::vector<cv::DMatch>& m
        ){
    m.clear();
    for(auto& m2 : m_knn){
        if(m2.size() < 1) continue;
        if(m2[0].distance >= maxd_) continue;
        if(m2.size() >= 2 && m2[0].distance >= lowe_ * m2[1].distance) continue;
        m.push_back( m2[0] );
    }
}

void Matcher::epifilter(
        const Frame& kf0, const Frame& kf1,
        const std::vector<cv::DMatch>& m_in,
        std::vector<cv::DMatch>& m_out
        ){
    std::vector<cv::Point2d> p0, p1;
    for(auto& m : m_in){
        p0.push_back( kf0.kpt[m.queryIdx]); //.pt );
        p1.push_back( kf1.kpt[m.trainIdx]); //.pt );
    }
    if (p0.size() <= 5 || p1.size() <= 5) return;

    cv::Mat msk;
    cv::findEssentialMat(p0, p1, 
            K_, cv::RANSAC, 0.999, 1.0, msk);

    m_out.clear();
    for(int i=0; i<msk.rows; ++i){
        if(!msk.at<char>(i)) continue;
        m_out.push_back(m_in[i]);
    }
}

void Matcher::match(
        const Frame& kf0,
        const Frame& kf1,
        std::vector<cv::DMatch>& match,
        bool cross,
        bool epicheck
        ){

    std::vector<cv::DMatch> mbuf;

    std::vector<cv::DMatch>& match0 = (epicheck? mbuf : match);

    if(cross){
        // bidirectional search
        std::vector<cv::DMatch> m01, m10;
        this->match(kf0, kf1, m01, false, false);
        this->match(kf1, kf0, m10, false, false);

        // fill match
        match0.clear();
        for(auto& m_fw : m01){
            bool found = false;
            for(auto& m_bw : m10){
                if(m_fw.queryIdx == m_bw.trainIdx &&
                        m_fw.trainIdx == m_bw.queryIdx){
                    found=true;
                    break;
                }
            }
            if(found) match0.push_back(m_fw);
        }
    }else{
        // initial match
        std::vector<std::vector<cv::DMatch>> m_knn;
        matcher_.knnMatch(kf0.dsc, kf1.dsc, m_knn, 2);

        // lowe + maxd filter
        filter(m_knn, match0);
    }

    // filter by epipolar constraint
    if(epicheck){ 
        epifilter(kf0, kf1, mbuf, match);
    }
}
