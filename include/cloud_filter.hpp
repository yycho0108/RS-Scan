#ifndef __CLOUD_FILTER_HPP__
#define __CLOUD_FILTER_HPP__

// pcl
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/covariance_sampling.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/keypoints/iss_3d.h>

struct CloudFilter{
    //pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> norm_filter;
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::PointNormal> norm_filter_iin;

    CloudFilter()
    {
        //norm_filter.setSearchMethod(tree);
        //norm_filter.setKSearch(8);

        norm_filter_iin.setNormalEstimationMethod (norm_filter_iin.AVERAGE_3D_GRADIENT);
        norm_filter_iin.setMaxDepthChangeFactor(0.02f);
        norm_filter_iin.setNormalSmoothingSize(10.0f);
    }

    template<typename T>
    void downsample_sub(
            const typename pcl::PointCloud<T>::ConstPtr& cld_in,
            typename pcl::PointCloud<T>::Ptr& cld_out,
            size_t scale=2
            ){
        // sub-sampling
        cld_out->is_dense = cld_in->is_dense;
        cld_out->width = cld_in->width / scale;
        cld_out->height = cld_in->height / scale;

        for( int ii = 0; ii < cld_in->height; ii+=scale){
            for( int jj = 0; jj < cld_in->width; jj+=scale ){
                cld_out->points.push_back(cld_in->points[ii*cld_in->width + jj]);
            }
        }
    }

    template<typename T>
    void downsample_vox(
            const typename pcl::PointCloud<T>::ConstPtr& cld_in,
            typename pcl::PointCloud<T>::Ptr& cld_out){
        // cloud --> downsampled cloud
        pcl::VoxelGrid<T> vox_filter;
        vox_filter.setLeafSize (0.025, 0.025, 0.025);
        vox_filter.setInputCloud (cld_in);
        vox_filter.filter (*cld_out);
    }

    void extract_keypoints(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr cld_in,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cld_out){

    }

    //void extract_features(
    //    const pcl::PointCloud<pcl::PointXYZ>::Ptr cld_in,
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cld_out){
    //}

    void downsample_cov(
            const pcl::PointCloud<pcl::PointNormal>::Ptr cld_in,
            pcl::PointCloud<pcl::PointNormal>::Ptr cld_out){

        pcl::CovarianceSampling<pcl::PointNormal, pcl::PointNormal> filter;
        filter.setInputCloud (cld_in);
        filter.setNormals (cld_in);
        filter.setNumberOfSamples(static_cast<unsigned int>(
                    cld_in->size ()) / 8); // 8x reduction

        double cnum = filter.computeConditionNumber ();
        //std::cout << cnum << std::endl;
        filter.filter(*cld_out);
    }

    template<typename T>
    void downsample_iss(
            const typename pcl::PointCloud<T>::ConstPtr& cld_in,
            typename pcl::PointCloud<T>::Ptr& cld_out
            ){
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal>);
        tree->setInputCloud( cld_in );

        pcl::ISSKeypoint3D<T, T> iss_detector;

        float model_resolution = 0.025;
        //iss_detector.setSearchMethod (tree);
        iss_detector.setSalientRadius (6 * model_resolution);
        iss_detector.setNonMaxRadius (4 * model_resolution);
        //iss_detector.setThreshold21 (0.975);
        //iss_detector.setThreshold32 (0.975);
        //iss_detector.setMinNeighbors (5);
        //iss_detector.setNumberOfThreads (4);
        iss_detector.setInputCloud (cld_in);
        iss_detector.compute (*cld_out);
        

        // transfer normals
        // pcl::copyPointCloud(*cld_in, *cld_out, idx);
        pcl::PointIndicesConstPtr idx = iss_detector.getKeypointsIndices ();
        for(int i=0; i<idx->indices.size(); ++i){
            cld_out->points[i].normal_x = cld_in->points[idx->indices[i]].normal_x;
            cld_out->points[i].normal_y = cld_in->points[idx->indices[i]].normal_y;
            cld_out->points[i].normal_z = cld_in->points[idx->indices[i]].normal_z;
        }
    }

    template<typename T>
    void inlier(
            const typename pcl::PointCloud<T>::ConstPtr& cld_in,
            typename pcl::PointCloud<T>::Ptr& cld_out
            ){
        pcl::StatisticalOutlierRemoval<T> of;
        // cloud --> filtered cloud
        of.setInputCloud(cld_in);
        of.setMeanK(100);
        of.setStddevMulThresh(1.4);
        of.filter(*cld_out);
    }

    void fill_xyz(
            const FrameData& kf,
            const cv::Mat& K,
            pcl::PointCloud<pcl::PointXYZ>::Ptr cld,
            float max_z=10.0
            ){
        // keyframe --> cloud

        // unroll params
        const float fx = K.at<float>(0, 0);
        const float fy = K.at<float>(1, 1);
        const float cx = K.at<float>(0, 2);
        const float cy = K.at<float>(1, 2);

        const int w = kf.img_.cols;
        const int h = kf.img_.rows;

        const float nan = std::numeric_limits<float>::quiet_NaN();

        // header
        cld->width    = w;
        cld->height   = h;
        cld->is_dense = false;
        cld->points.resize(cld->width*cld->height);

        // fill cloud
        for(size_t i=0; i<h; ++i){
            for(size_t j=0; j<w; ++j){
                float z = kf.d_img_.at<float>(i,j);
                auto& p = cld->points[i*w+j];

                if(z<0 || z>max_z){
                    // mark as invalid
                    p.x=p.y=p.z=nan;
                }else{
                    p.z = z;
                    p.x = (j-cx) * (z / fx);
                    p.y = (i-cy) * (z / fy);
                }

            }
        }
    }

    void fill_rgb(
            const FrameData& kf,
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld
            ){
        const int w = kf.img_.cols;
        const int h = kf.img_.rows;

        // fill cloud
        for(size_t i=0; i<h; ++i){
            for(size_t j=0; j<w; ++j){
                const cv::Vec3b& bgr = kf.img_.at<cv::Vec3b>(i, j);
                auto& p = cld->points[i*w+j];
                p.b = bgr[0];
                p.g = bgr[1];
                p.r = bgr[2];
            }
        }
    }

    void fill_xyzrgb(
            const FrameData& kf,
            const cv::Mat& K,
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld,
            float max_z=10.0){

        // keyframe --> cloud

        // unroll params
        const float fx = K.at<float>(0, 0);
        const float fy = K.at<float>(1, 1);
        const float cx = K.at<float>(0, 2);
        const float cy = K.at<float>(1, 2);

        const int w = kf.img_.cols;
        const int h = kf.img_.rows;

        const float nan = std::numeric_limits<float>::quiet_NaN();

        // header
        cld->width    = w;
        cld->height   = h;
        cld->is_dense = false;
        cld->points.resize(cld->width*cld->height);

        // fill cloud
        for(size_t i=0; i<h; ++i){
            for(size_t j=0; j<w; ++j){
                // xyz
                float z = kf.d_img_.at<float>(i,j);
                auto& p = cld->points[i*w+j];

                if(z<0 || z>max_z){
                    // mark as invalid
                    p.x=p.y=p.z=nan;
                }else{
                    p.z = z;
                    p.x = (j-cx) * (z / fx);
                    p.y = (i-cy) * (z / fy);
                }

                // color
                const cv::Vec3b& bgr = kf.img_.at<cv::Vec3b>(i, j);
                p.b = bgr[0];
                p.g = bgr[1];
                p.r = bgr[2];
            }
        }
    }

    //void fill_xyzrgb(
    //        const FrameData& kf,
    //        const cv::Mat& K,
    //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld,
    //        float max_z=10.0
    //        ){

    //}

    void fill_normal(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr cld_in,
            pcl::PointCloud<pcl::PointNormal>::Ptr cld_out){
        //norm_filter.setRadiusSearch(0.05);
        //norm_filter.setInputCloud (cld_in);
        //norm_filter.compute (*cld_out);
        norm_filter_iin.setInputCloud(cld_in);
        norm_filter_iin.compute(*cld_out);
        pcl::copyPointCloud (*cld_in, *cld_out);
    }

    void calc(
            const FrameData& kf0,
            const FrameData& kf1,
            Eigen::Isometry3d& T){

    }
};


#endif
