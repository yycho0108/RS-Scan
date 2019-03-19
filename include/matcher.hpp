#ifndef __MATCHER_HPP__
#define __MATCHER_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "frame.hpp"

struct Matcher{
    cv::Mat K_;
    float lowe_;
    float maxd_;
    cv::FlannBasedMatcher matcher_;

    Matcher(const cv::Mat& K, float lowe=0.8, float maxd=64.0);
    void filter(
            const std::vector<std::vector<cv::DMatch>>& m_knn,
            std::vector<cv::DMatch>& m
            );
    void epifilter(
            const Frame& kf0, const Frame& kf1,
            const std::vector<cv::DMatch>& m_in,
            std::vector<cv::DMatch>& m_out
            );
    void match(
            const Frame& kf0,
            const Frame& kf1,
            std::vector<cv::DMatch>& match,
            bool cross=false,
            bool epicheck=false
            );
};

#endif
