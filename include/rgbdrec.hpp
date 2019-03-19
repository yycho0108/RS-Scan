#ifndef __RGBDREC_HPP__
#define __RGBDREC_HPP__

#include "types.hpp"
#include "tracker.hpp"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

class KalmanFilter{};

class RGBDRec{
    // parameters
    cv::Mat K_, Ki_;

    // data
    std::vector<FrameData> db_;

    // handles
    KalmanFilter mkf_;
    Tracker trk_;
	cv::Ptr<cv::FeatureDetector> det_;
	cv::Ptr<cv::DescriptorExtractor> des_;

    void prefill_frame(
            const cv::Mat& dmap,
            const cv::Mat& cmap,
            const double stamp,
            FrameData& kf){

    }

    void apply_motion(const FrameData& kf0, FrameData& kf1);

    bool icp_f2f(
            const FrameData& kf0,
            const FrameData& kf1,
            const std::vector<cv::DMatch>& m01,
            Eigen::Isometry3d& T
            ){
        // unroll data
        float fx = K_.at<float>(0, 0);
        float fy = K_.at<float>(1, 1);
        float cx = K_.at<float>(0, 2);
        float cy = K_.at<float>(1, 2);

        // create point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld1 (new pcl::PointCloud<pcl::PointXYZ>);
        for(auto& cld : {cld0, cld1}){
            cld->width    = m01.size();
            cld->height   = 1;
            cld->is_dense = false;
            cld->points.resize(cld->width * cld->height);
        }

        // Fill in the CloudIn data
        // TODO: maybe better as a method, kf0.point3d(i0)
        // TODO: apply pre-filled pose

        for(size_t i=0; i<m01.size(); ++i){
            // parsing elements
            auto& m = m01[i];

            const int i0 = m.queryIdx;
            const int i1 = m.trainIdx;

            float z0 = kf0.depth(i0);
            float x0 = z0 * (kf0.kpt_[i0].pt.x / fx - cx*fx); // K^{-1}.h(x)
            float y0 = z0 * (kf0.kpt_[i0].pt.y / fy - cy*fy);
            cld0->points[i].x = x0;
            cld0->points[i].y = y0;
            cld0->points[i].z = z0;

            float z1 = kf1.depth(i1);
            float x1 = z1 * (kf1.kpt_[i1].pt.x / fx - cx*fx); // K^{-1}.h(x)
            float y1 = z1 * (kf1.kpt_[i1].pt.y / fy - cy*fy);
            cld1->points[i].x = x1;
            cld1->points[i].y = y1;
            cld1->points[i].z = z1;
        }

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(cld0);
        icp.setInputTarget(cld1);
        pcl::PointCloud<pcl::PointXYZ> Final;
        icp.align(Final);

        if( !icp.hasConverged() ) return false;
        std::cout << "has converged:" << icp.hasConverged() << " score: " <<
            icp.getFitnessScore() << std::endl;

        // TODO : if match failed, use pixel indices as NN guess
        // with pcl::IterativeClosestPointWithNormals<>

        std::cout << icp.getFinalTransformation() << std::endl;
        T = icp.getFinalTransformation().cast<double>();
    }

    bool process_frame(
            const cv::Mat& dmap,
            const cv::Mat& cmap,
            double stamp
            ){

        // 0) process data
        FrameData kf1;
        prefill_frame(dmap, cmap, stamp, kf1);
        if(! db_.size() > 0){
            db_.push_back( std::move(kf1) );
            return true;
        }

        // 1) motion-based pose guess
        FrameData& kf0 = db_.back();
        this->apply_motion(kf0, kf1);

        // 2) frame-to-frame refinement
        std::vector<cv::DMatch> m01;
        this->trk_.track(kf0, kf1, m01);

    }

};

#endif
