#ifndef __RGBDREC_HPP__
#define __RGBDREC_HPP__

#include "types.hpp"
#include "tracker.hpp"
#include "cloud_filter.hpp"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/visualization/pcl_visualizer.h>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

cv::RNG rng(12345);
// This is a tutorial so we can afford having global variables 
//our visualizer
pcl::visualization::PCLVisualizer *p;
//its left and right viewports
int vp_1, vp_2;

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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld_map_;
    std::vector<FrameData> db_;

    // handles
    KalmanFilter mkf_;
    Tracker trk_;
    cv::Ptr<cv::FeatureDetector> det_;
    cv::Ptr<cv::DescriptorExtractor> des_;
    CloudFilter cf_;

    RGBDRec():
    cld_map_(new pcl::PointCloud<pcl::PointXYZRGB>)
    {
        det_ = cv::ORB::create();
        des_ = cv::ORB::create();
        K_ = cv::Mat(3, 3, CV_32F);
        Ki_ = cv::Mat(3, 3, CV_32F);

        int argc = 1;
        char* argv[] = {"main"};
        
        p = new pcl::visualization::PCLVisualizer (argc, argv,
                "Pairwise Incremental Registration example");
        p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
        p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);


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

    bool icp_f2f_shot(
            const FrameData& kf0,
            const FrameData& kf1,
            Eigen::Isometry3d& T
            ){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld0 (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld1 (new pcl::PointCloud<pcl::PointXYZRGB>);

        // fill cloud
        cf_.fill_xyzrgb(kf0, K_, cld0);
        cf_.fill_xyzrgb(kf1, K_, cld1);

        //pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> des;
    }

    bool icp_f2f_full(
            const FrameData& kf0,
            const FrameData& kf1,
            Eigen::Isometry3d& T
            ){

        // create intermediate point clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld1 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0_s (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld1_s (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld0_n (new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld1_n (new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld0_f(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld1_f(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld0_d(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld1_d(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld0_i(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr cld1_i(new pcl::PointCloud<pcl::PointNormal>);

        // fill cloud
        cf_.fill_xyz(kf0, K_, cld0);
        cf_.fill_xyz(kf1, K_, cld1);

        // downsample 4x
        cf_.downsample_sub<pcl::PointXYZ>(cld0, cld0_s, 4);
        cf_.downsample_sub<pcl::PointXYZ>(cld1, cld1_s, 4);

        // apply input transform
        pcl::transformPointCloud (*cld1_s, *cld1_s, T.cast<float>());

        // fill normals (IIN)
        cf_.fill_normal(cld0_s, cld0_n);
        cf_.fill_normal(cld1_s, cld1_n);

        // statistical inlier
        cf_.inlier<pcl::PointNormal>(cld0_n, cld0_i);
        cf_.inlier<pcl::PointNormal>(cld1_n, cld1_i);

        // cleanup
        std::vector<int> tmp;
        pcl::removeNaNFromPointCloud(*cld0_i, *cld0_f, tmp);
        cld0_i.swap(cld0_f);
        pcl::removeNaNNormalsFromPointCloud(*cld0_i, *cld0_f, tmp);

        pcl::removeNaNFromPointCloud(*cld1_i, *cld1_f, tmp);
        cld1_i.swap(cld1_f);
        pcl::removeNaNNormalsFromPointCloud(*cld1_i, *cld1_f, tmp);

        //std::cout << cld0_n->is_dense << ',' << cld1_n->is_dense << std::endl;
        //std::cout << cld0_f->is_dense << ',' << cld1_f->is_dense << std::endl;

        // downsample
        // downsample_vox<> / ...
        cf_.downsample_iss<pcl::PointNormal>(cld0_f, cld0_d);
        cf_.downsample_iss<pcl::PointNormal>(cld1_f, cld1_d);

        //for(auto& p : cld0_i->points){
        //    if(!pcl::isFinite(p)){
        //        std::cout << "(p0) finite check failed!" << std::endl;
        //    }

        //    if(!pcl::isFinite(p.normal)){
        //        std::cout << "(n0) finite check failed!" << std::endl;
        //    }
        //}

        //for(auto& p : cld1_i->points){
        //    if(!pcl::isFinite(p)){
        //        std::cout << "(p1) finite check failed!" << std::endl;
        //    }

        //    if(!pcl::isFinite(p.normal)){
        //        std::cout << "(n1) finite check failed!" << std::endl;
        //    }
        //}

        pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
        icp.setTransformationEpsilon (1e-8);
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm
        // Note: adjust this based on the size of your datasets
        icp.setMaxCorrespondenceDistance (0.1);
        icp.setMaximumIterations (50);

        // Set the point representation
        //icp.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

        icp.setInputSource (cld1_d); // == T * cld1_n
        icp.setInputTarget (cld0_d);

        pcl::PointCloud<pcl::PointNormal> Final;
        //std::cout << "ICP ALIGN START" << std::endl;
        icp.align(Final);
        //std::cout << "ICP ALIGN DONE" << std::endl;

        // viz f2f results
        p->removePointCloud ("source");
        p->removePointCloud ("target");
        p->removePointCloud ("map"); 
        p->removePointCloud ("current");

        PointCloudColorHandlerCustom<pcl::PointNormal> cloud_tgt_h (cld0_i, 0, 255, 0);
        PointCloudColorHandlerCustom<pcl::PointNormal> cloud_src_h (cld1_i, 255, 0, 0);
        p->addPointCloud (cld0_i, cloud_tgt_h, "target", vp_2);
        p->addPointCloud (cld1_i, cloud_src_h, "source", vp_2);

        // viz aggregate results
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0_T (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud (*cld0, *cld0_T, kf0.pose_.cast<float>() );

        pcl::PointCloud<pcl::PointNormal>::Ptr cld0_Ti (new pcl::PointCloud<pcl::PointNormal>);
        pcl::transformPointCloud (*cld0_i, *cld0_Ti, kf0.pose_.cast<float>() );

        // add + downsample
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*cld0_T, *cld_tmp); // fill xyz
        cf_.fill_rgb(kf0, cld_tmp); // fill rgb

        *cld_tmp += *cld_map_;
        //cf_.downsample(cld_tmp, cld_map_);
        pcl::VoxelGrid<pcl::PointXYZRGB> vox_filter;
        vox_filter.setLeafSize (0.025, 0.025, 0.025);
        vox_filter.setInputCloud (cld_tmp);
        vox_filter.filter (*cld_map_);

        PointCloudColorHandlerCustom<pcl::PointNormal> cld0_T_h (cld0_Ti, 255, 0, 0);
        //PointCloudColorHandlerCustom<pcl::PointXYZRGB> cld_map_h (cld0_T, 0, 0, 255);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cld_map_h(cld_map_);

        p->addPointCloud (cld_map_, cld_map_h, "map", vp_1);
        p->addPointCloud (cld0_Ti, cld0_T_h, "current", vp_1);

        p->setPointCloudRenderingProperties (
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                3, "map");

        //PCL_INFO ("Press q to continue the registration.\n");
        p->spinOnce ();
        p->removePointCloud ("source"); 
        p->removePointCloud ("target");
        p->removePointCloud ("map"); 
        p->removePointCloud ("current");

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

        T.matrix() = icp.getFinalTransformation().cast<double>() * T.matrix();
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
        // create intermediate point clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld0 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld1 (new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cld0_i(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cld1_i(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cld0_d(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cld1_d(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointNormal>::Ptr cld0_n (new pcl::PointCloud<pcl::PointNormal>);
        //pcl::PointCloud<pcl::PointNormal>::Ptr cld1_n (new pcl::PointCloud<pcl::PointNormal>);

        // fill cloud
        cf_.fill_xyz(kf0, K_, cld0);
        cf_.fill_xyz(kf1, K_, cld1);

        // assume tracking mostly solved corr ...
        // pcl::CorrespondenceEstimation<?,?> est;
        pcl::CorrespondencesPtr corr0(new pcl::Correspondences),
            corr1(new pcl::Correspondences);
        corr0->reserve(m01.size());

        const int w = kf0.img_.cols;
        for(auto& m : m01){

            int j0 = pt0[m.queryIdx].x;
            int i0 = pt0[m.queryIdx].y;

            int j1 = pt1[m.trainIdx].x;
            int i1 = pt1[m.trainIdx].y;

            if (!std::isfinite(cld0->points[i0*w+j0].z))
                continue;

            if (!std::isfinite(cld1->points[i1*w+j1].z))
                continue;

            corr0->emplace_back(
                    i1*w+j1, i0*w+j0, 
                    m.distance);
        }

        pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > sac;
        sac.setInputCloud( cld1 );
        sac.setTargetCloud( cld0 );
        sac.setInlierThreshold( 0.05 );
        sac.setMaxIterations(100);
        sac.setInputCorrespondences( corr0 );
        sac.getCorrespondences( *corr1 );

        // ..-> T * cld1

        T.matrix() = sac.getBestTransformation().cast<double>();
        return true;
#if 0

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
#endif
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

        cv::Mat viz = drawMatches(kf0, kf1, pt0, pt1, m01);
        cv::imshow("viz", viz);

        Eigen::Isometry3d T01;
        T01.setIdentity();
        bool icp_suc = icp_f2f(
                kf0, kf1, m01,
                pt0, pt1,
                T01);
        if(!icp_suc){
            T01.setIdentity();
        }

        bool icp_suc2 = icp_f2f_full(kf0, kf1, T01);
        //bool icp_suc2=false;

        if(icp_suc || icp_suc2){
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
