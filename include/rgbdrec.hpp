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

cv::RNG rng(12345);

class KalmanFilter{};
cv::Mat drawMatches(
        const FrameData& kf0,
        const FrameData& kf1,
        const std::vector<cv::Point2f>& pt0,
        const std::vector<cv::Point2f>& pt1,
        const std::vector<cv::DMatch>& m01
        ){

    cv::Mat viz_img;
    cv::hconcat(kf0.img_, kf1.img_, viz_img);
    for(auto& m : m01){
        auto vpt0 = pt0[ m.queryIdx ];
        auto vpt1_0 = pt1[ m.trainIdx ];
        auto vpt1 = cv::Point(vpt1_0.x+kf0.img_.cols, vpt1_0.y);
        cv::line(viz_img, vpt0, vpt1, cv::Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), 5);
    }

    for(auto& m : m01){
        auto vpt0 = pt0[ m.queryIdx ];
        auto vpt1_0 = pt1[ m.trainIdx ];
        auto vpt1 = cv::Point(vpt1_0.x+kf0.img_.cols, vpt1_0.y);
        cv::circle(viz_img, vpt0, 3, cv::Scalar(255,0,0), 3);
        cv::circle(viz_img, vpt1, 3, cv::Scalar(255,0,0), 3);
    }

    return viz_img;
}

struct RGBDRec{
    // parameters
    cv::Mat K_, Ki_;

    // data
    std::vector<FrameData> db_;

    // handles
    KalmanFilter mkf_;
    Tracker trk_;
	cv::Ptr<cv::FeatureDetector> det_;
	cv::Ptr<cv::DescriptorExtractor> des_;

    RGBDRec(){
        det_ = cv::ORB::create();
        des_ = cv::ORB::create();
        K_ = cv::Mat(3, 3, CV_32F);
        Ki_ = cv::Mat(3, 3, CV_32F);
    }

    void set_intrinsic(
            float fx, float fy,
            float cx, float cy){
        K_.at<float>(0,0) = fx;
        K_.at<float>(1,1) = fy;

        K_.at<float>(0,2) = cx;
        K_.at<float>(1,2) = cy;

        K_.at<float>(2,2) = 1.0;

        Ki_ = K_.inv();
    }

    void prefill_frame(
            const cv::Mat& dmap,
            const cv::Mat& cmap,
            const double stamp,
            FrameData& kf){
        // TODO : rectify frames????

        // copy data
        kf.img_ = cmap;
        kf.d_img_ = dmap;
        kf.stamp_ = stamp;

        // set defaults
        kf.pose_.setIdentity();

        // fill processed features 
        cv::cvtColor(cmap, kf.gray_, cv::COLOR_BGR2GRAY);
        det_->detectAndCompute(kf.img_, cv::Mat(),
                kf.kpt_,
                kf.dsc_);

    }

    void apply_motion(const FrameData& kf0, FrameData& kf1){}

    bool icp_f2f(
            const FrameData& kf0,
            const FrameData& kf1,
            const std::vector<cv::DMatch>& m01,
            const std::vector<cv::Point2f>& pt0,
            const std::vector<cv::Point2f>& pt1,
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

            float z0 = kf0.d_img_.at<float>(
                    pt0[i0].y,
                    pt0[i0].x);
            float x0 = z0 * (pt0[i0].x / fx - cx/fx); // K^{-1}.h(x)
            float y0 = z0 * (pt0[i0].y / fy - cy/fy);
            cld0->points[i].x = x0;
            cld0->points[i].y = y0;
            cld0->points[i].z = z0;

            float z1 = kf1.d_img_.at<float>(
                    pt1[i1].y,
                    pt1[i1].x);
            float x1 = z1 * (pt1[i1].x / fx - cx/fx); // K^{-1}.h(x)
            float y1 = z1 * (pt1[i1].y / fy - cy/fy);
            cld1->points[i].x = x1;
            cld1->points[i].y = y1;
            cld1->points[i].z = z1;
            //std::cout << z0 << ',' << z1 << std::endl;
        }

        // get initial estimate from RGB alignment
        pcl::registration::TransformationEstimationSVD<
            pcl::PointXYZ,
            pcl::PointXYZ,
            float //?
            > est;
        Eigen::Matrix<float, 4, 4> Tf;
        est.estimateRigidTransformation(*cld1, *cld0, Tf);

        // Want : T1 = Tf * T0
        // Tf = T1 * T0^{-1}

        // T1 * cld1 = Tf * T0 * cld0

        // Tf * cld1 == cld0;

        // T0 * cld0 = T1 * cld1, Tf = T0^{-1}*T1
        // T1 = Tf
        // want : T1 | {T0, Tf}

        T = Tf.cast<double>();

        // additional ICP?
        ////pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        ////icp.setInputSource(cld0);
        ////icp.setInputTarget(cld1);
        ////pcl::PointCloud<pcl::PointXYZ> Final;
        ////icp.align(Final);

        //if( !icp.hasConverged() ) return false;
        //std::cout << "has converged:" << icp.hasConverged() << " score: " <<
        //    icp.getFitnessScore() << std::endl;

        //// TODO : if match failed, use pixel indices as NN guess
        //// with pcl::IterativeClosestPointWithNormals<>

        //std::cout << icp.getFinalTransformation() << std::endl;
        //T = icp.getFinalTransformation().cast<double>();
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
        std::vector<cv::Point2f> pt0, pt1;
        std::vector<cv::DMatch> m01;
        this->trk_.track(kf0, kf1, pt0, pt1, m01);

        //cv::Mat viz = drawMatches(kf0, kf1, pt0, pt1, m01);
        //cv::imshow("viz", viz);

        // yay!
        Eigen::Isometry3d T01;
        bool icp_suc = icp_f2f(
                kf0, kf1, m01,
                pt0, pt1,
                T01);
        if(icp_suc){
            std::cout << "ICP SUC" << std::endl;
            std::cout << T01.matrix() << std::endl;
        }

        // save frame
        db_.push_back( std::move(kf1) );
        return true;
    }

};

#endif
