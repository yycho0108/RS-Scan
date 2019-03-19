#ifndef __TYPES_HPP__
#define __TYPES_HPP__

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <Eigen/Geometry>

struct Twist{
    float vx, vy, vz;
    float wx, wy, wz;
};

struct FrameData{
    // visual data
    cv::Mat img_;
    cv::Mat d_img_;
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;
    std::vector<cv::KeyPoint> kpt_;
    cv::Mat dsc_;

    // positional data
    Eigen::Isometry3d pose_;
    Twist twist_;
    double stamp_;

    float depth(size_t idx) const{
        // TODO : support indexing radius
        int x=kpt_[idx].pt.x;
        int y=kpt_[idx].pt.y;
        return d_img_.at<float>(y, x);
    }
};

#endif
