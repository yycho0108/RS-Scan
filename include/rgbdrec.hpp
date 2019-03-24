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
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>


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

template<typename T>
void fill_point(T& pt,
        float x, float y,
        float z,
        float fx, float fy,
        float cx, float cy){
}

struct ICPPipeline{
    pcl::VoxelGrid<pcl::PointXYZ> vox_filter;

    void downsample(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr cld_in,
            pcl::PointCloud<pcl::PointXYZ>::Ptr cld_out){
        // cloud --> downsampled cloud
    }

    void fill_xyz(
            const FrameData& kf,
            pcl::PointCloud<pcl::PointXYZ>::Ptr cld){
        // keyframe --> cloud
    }

    void fill_normal(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr cld_in,
            pcl::PointCloud<pcl::PointNormal>::Ptr cld_out){
    }

    void calc(
            const FrameData& kf0,
            const FrameData& kf1,
            Eigen::Isometry3d& T){

    }
};

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
        kf.img_ = cmap.clone();
        kf.d_img_ = dmap.clone();
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

    bool icp_f2f_full(
            const FrameData& kf0,
            const FrameData& kf1,
            Eigen::Isometry3d& T
            ){
        // unroll data
        const float fx = K_.at<float>(0, 0);
        const float fy = K_.at<float>(1, 1);
        const float cx = K_.at<float>(0, 2);
        const float cy = K_.at<float>(1, 2);

        const int w = kf0.img_.cols;
        const int h = kf0.img_.rows;
        
        const float nan = std::numeric_limits<float>::quiet_NaN();
        // create point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld1 (new pcl::PointCloud<pcl::PointXYZ>);
        for(auto& cld : {cld0, cld1}){
            cld->width    = w*h;
            cld->height   = 1;
            cld->is_dense = false;
            cld->points.resize(cld->width*cld->height);
        }

        // fill cloud
        for(size_t i=0; i<h; ++i){
            for(size_t j=0; j<w; ++j){
                float z0 = kf0.d_img_.at<float>(i,j);
                float z1 = kf1.d_img_.at<float>(i,j);
                auto& p0 = cld0->points[i*w+j];
                auto& p1 = cld1->points[i*w+j];

                if(z0<0 || z0>10.0){
                    // mark as invalid
                    p0.x=p0.y=p0.z=nan;
                }else{
                    p0.z = z0;
                    p0.x = (j - cx) * (z0 / fx);
                    p0.y = (i - cy) * (z0 / fy);
                }

                if(z1<0 || z1>10.0){
                    // mark as invalid
                    p1.x=p1.y=p1.z=nan;
                }else{
                    p1.z = z1;
                    p1.x = (j - cx) * (z1 / fx);
                    p1.y = (i - cy) * (z1 / fy);
                }
            }
        }
        // downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0_d(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld1_d(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> grid;
        grid.setLeafSize (0.025, 0.025, 0.025);
        grid.setInputCloud (cld0);
        grid.filter (*cld0_d);

        grid.setInputCloud (cld1);
        grid.filter (*cld1_d);

        // Compute surface normals and curvature
        //std::cout << "NORM" << std::endl;
        pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_cld0 (new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_cld1 (new pcl::PointCloud<pcl::PointNormal>);

        pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> norm_est;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        norm_est.setSearchMethod (tree);
        norm_est.setKSearch (30);

        //std::cout << "NORM-1" << std::endl;
        norm_est.setInputCloud (cld0_d);
        norm_est.compute (*points_with_normals_cld0);
        pcl::copyPointCloud (*cld0_d, *points_with_normals_cld0);

        //std::cout << "NORM-2" << std::endl;
        norm_est.setInputCloud (cld1_d);
        norm_est.compute (*points_with_normals_cld1);
        pcl::copyPointCloud (*cld1_d, *points_with_normals_cld1);

        //std::cout << cld0->points.size() << std::endl;
        //std::cout << cld1->points.size() << std::endl;
        //std::cout << cld0_d->points.size() << std::endl;
        //std::cout << cld1_d->points.size() << std::endl;
        //std::cout << points_with_normals_cld0->points.size() << std::endl;
        //std::cout << points_with_normals_cld1->points.size() << std::endl;
        
        pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
        icp.setTransformationEpsilon (1e-6);
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm
        // Note: adjust this based on the size of your datasets
        icp.setMaxCorrespondenceDistance (0.025);
        // Set the point representation
        //icp.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

        icp.setInputSource (points_with_normals_cld1);
        icp.setInputTarget (points_with_normals_cld0);

        pcl::PointCloud<pcl::PointNormal> Final;
        //std::cout << "ICP ALIGN START" << std::endl;
        icp.align(Final);
        //std::cout << "ICP ALIGN DONE" << std::endl;

        if( !icp.hasConverged() ){
            std::cout << "ICP FAIL" << std::endl;
            return false;
        }
        //std::cout << "has converged:" << icp.hasConverged() << " score: " <<
        //    icp.getFitnessScore() << std::endl;

        // TODO : if match failed, use pixel indices as NN guess
        // with pcl::IterativeClosestPointWithNormals<>

        //std::cout << icp.getFinalTransformation() << std::endl;

        // validation
        // pcl::PointCloud<pcl::PointXYZ>::Ptr tx_cld0 (new pcl::PointCloud<pcl::PointXYZ> ());
        // pcl::PointCloud<pcl::PointXYZ>::Ptr tx_cld1 (new pcl::PointCloud<pcl::PointXYZ> ());
        // pcl::transformPointCloud (*cld0, *tx_cld0, kf0.pose_);
        // pcl::transformPointCloud (*cld1, *tx_cld1, kf0.pose_*icp.getFinalTransformation());

        T = icp.getFinalTransformation().cast<double>();
        return true;
    }


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

        size_t j=0;
        for(size_t i=0; i<m01.size(); ++i){
            // parsing elements
            auto& m = m01[i];

            const int i0 = m.queryIdx;
            const int i1 = m.trainIdx;

            float z0 = kf0.d_img_.at<float>(
                    pt0[i0].y,
                    pt0[i0].x);

            float z1 = kf1.d_img_.at<float>(
                    pt1[i1].y,
                    pt1[i1].x); // TODO : better query

            if( z0<0 || z0>10.0 || z1<0 || z1>10.0) continue;
            if(! (std::isfinite(z0) && std::isfinite(z1)) ) continue;

            float x0 = z0 * (pt0[i0].x / fx - cx/fx); // K^{-1}.h(x)
            float y0 = z0 * (pt0[i0].y / fy - cy/fy);
            cld0->points[j].x = x0;
            cld0->points[j].y = y0;
            cld0->points[j].z = z0;

            float x1 = z1 * (pt1[i1].x / fx - cx/fx); // K^{-1}.h(x)
            float y1 = z1 * (pt1[i1].y / fy - cy/fy);
            cld1->points[j].x = x1;
            cld1->points[j].y = y1;
            cld1->points[j].z = z1;
            //std::cout << z0 << ',' << z1 << std::endl;

            ++j;
        }

        for(auto& cld : {cld0, cld1}){
            cld->width = j;
            cld->height   = 1;
            cld->is_dense = false;
            cld->points.resize(j);
        }

        // get initial estimate from RGB alignment
        pcl::registration::TransformationEstimationSVD<
            pcl::PointXYZ,
            pcl::PointXYZ,
            float //?
            > est;
        Eigen::Matrix<float, 4, 4> Tf;
        est.estimateRigidTransformation(*cld1, *cld0, Tf);
        // Tf * cld1 = cld0
        // T0^{-1} * T1 * cld1 = cld0
        // Tf = T0^{-1} * T1
        // T1 = T0 * Tf
        T.matrix() = Tf.cast<double>();
        return true;

        //// additional ICP?
        //pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        //icp.setInputSource(cld1);
        //icp.setInputTarget(cld0);
        //pcl::PointCloud<pcl::PointXYZ> Final;
        //icp.align(Final);

        //if( !icp.hasConverged() ) return false;
        ////std::cout << "has converged:" << icp.hasConverged() << " score: " <<
        ////    icp.getFitnessScore() << std::endl;

        //// TODO : if match failed, use pixel indices as NN guess
        //// with pcl::IterativeClosestPointWithNormals<>

        ////std::cout << icp.getFinalTransformation() << std::endl;

        //// validation
        //// pcl::PointCloud<pcl::PointXYZ>::Ptr tx_cld0 (new pcl::PointCloud<pcl::PointXYZ> ());
        //// pcl::PointCloud<pcl::PointXYZ>::Ptr tx_cld1 (new pcl::PointCloud<pcl::PointXYZ> ());
        //// pcl::transformPointCloud (*cld0, *tx_cld0, kf0.pose_);
        //// pcl::transformPointCloud (*cld1, *tx_cld1, kf0.pose_*icp.getFinalTransformation());

        //T = icp.getFinalTransformation().cast<double>();
        return true;
    }

    bool process_frame(
            const cv::Mat& dmap,
            const cv::Mat& cmap,
            double stamp
            ){
        // 0) process data
        FrameData kf1;
        prefill_frame(dmap, cmap, stamp, kf1);
        if(db_.size() <= 0){
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

        //bool icp_suc = icp_f2f_full(kf0, kf1, T01);

        if(icp_suc){
            //std::cout << "ICP SUC" << std::endl;
            //std::cout << T01.matrix() << std::endl;

            //kf1.pose_ = T01 * kf0.pose_;
            kf1.pose_ = kf0.pose_*T01;

            // save frame
            db_.push_back( std::move(kf1) );
        }else{
            // TODO : implement recovery and such
            // TODO : begin submap
            return false;
        }
        return true;
    }

};

#endif
