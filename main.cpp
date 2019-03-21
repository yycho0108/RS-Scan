//#include <librealsense/rs.hpp>
//#include <librealsense/rscore.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "kinectfusion.h"
#include "data_types.h"

#include "types.hpp"
#include "rgbdrec.hpp"

#include <iostream>
#include <fstream>

#include <algorithm>
#include <memory>
#include <cstring>

bool profile_changed(
		const std::vector<rs2::stream_profile>& cprof,
		const std::vector<rs2::stream_profile>& pprof){

    for (auto&& sp : pprof)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(cprof), std::end(cprof),
				[&sp](const rs2::stream_profile& current_sp) {
				return sp.unique_id() == current_sp.unique_id();
				}
				);
        if (itr == std::end(cprof)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

void set_color(rs2::video_frame& vf, cv::Mat& out){
	int w = vf.get_width();
	int h = vf.get_height();
	int d = vf.get_bytes_per_pixel(); // depth

	out.create(h, w, CV_8UC(d) );
	uint8_t* data = (uint8_t*)vf.get_data();
	//(pixy * texture_width + pixx) * num_colors

	for(int y=0; y<h; ++y){
		for(int x=0; x<w; ++x){
			int offset = (y*w+x)*d;

			// TODO : rgb vs. bgr?
			//out.at<cv::Vec3b>(y,x) = {data[offset+2],data[offset+1],data[offset]};
			// kinectfusion requires rgb for some reason
			out.at<cv::Vec3b>(y,x) = {data[offset],data[offset+1],data[offset+2]};
		}
	}
}

void set_depth(rs2::depth_frame& df, float depth_scale, cv::Mat& out){
	// stat frame data

    //uint16_t* data = reinterpret_cast<uint16_t*>(const_cast<void*>(df.get_data()));
    int w = df.get_width();
    int h = df.get_height();

    cv::Mat tmp{cv::Size {w, h},
                          CV_16UC1,
                          const_cast<void*>(df.get_data()),
                          cv::Mat::AUTO_STEP};
    tmp.convertTo(out, CV_32FC1, depth_scale);

	//// ensure output matrix is formatted correctly
	//out.create(h, w, CV_32FC1);

	//// TODO : maybe more efficient if data is copied over directly
	//// fill in data
    //for (int y = 0; y < h; y++)
    //{
    //    auto depth_pixel_index = y * w;
    //    for (int x = 0; x < w; x++, ++depth_pixel_index)
    //    {
    //        // Get the depth value of the current pixel
	//		// NOTE : trying mm
	//		out.at<float>(y, x) = depth_scale * data[depth_pixel_index];
    //    }
    //}
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}

//struct CameraParams{
//	float w, h;
//	float fx, fy;
//	float cx, cy;
//	float d[5];
//
//	CameraParams(){};
//	CameraParams(
//			float w, float h,
//			float fx,float fy,
//			float cx,float cy):
//		w(w),h(h),fx(fx),fy(fy),cx(cx),cy(cy){}
//};


class RS2{
	// rs2 handles
	rs2::pipeline pl_;
	rs2::pipeline_profile prof_;

	// kf handles
	kinectfusion::GlobalConfiguration kf_cfg_;
	kinectfusion::CameraParameters kf_params_;
	std::shared_ptr<kinectfusion::Pipeline> kf_pl_;

    // custom KF
    RGBDRec rec;

	// data cache 
	float depth_scale_ = 1.0f;
	rs2_stream align_to_;
	std::shared_ptr<rs2::align> align_;
    double stamp_;
	cv::Mat dmap, cmap;

  public:
	RS2(){
		kf_cfg_.voxel_scale = 2.f;
		kf_cfg_.init_depth = 700.0f; // ???
		kf_cfg_.distance_threshold = 10.0f;
		kf_cfg_.angle_threshold = 20.0f;
		kf_cfg_.depth_cutoff_distance = 2000.0f;
		start();
	};

	~RS2(){
		stop();
	}

	void start(){
		rs2::config cfg{};
        cfg.enable_device_from_file("/home/jamie/Documents/20190321_081722.bag");

		//cfg.disable_all_streams();
		//cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
		//cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

		prof_ = pl_.start(cfg);

        /*
        if( cfg_.size() > 0 ){
            auto amd = pl_.get_active_profile().get_device().as<rs400::advanced_mode>();
            std::ifstream t("/home/jamie/Documents/ShortRangePreset.json");
            std::string json_content((std::istreambuf_iterator<char>(t)),
                            std::istreambuf_iterator<char>());
            amd.load_json(json_content);
        }
        prof_ = pl_.get_active_profile();
        */

		update_profile(true);
		update_params(); // only needs to be performed once (hopefully)

		kf_pl_.reset(new kinectfusion::Pipeline {kf_params_, kf_cfg_ });
	}

	void stop(){
		pl_.stop();
	}

	void update_params(){
		auto intrinsic = prof_.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

		kf_params_.principal_x = intrinsic.ppx;
		kf_params_.principal_y = intrinsic.ppy;
		kf_params_.image_width  = intrinsic.width;
		kf_params_.image_height  = intrinsic.height;
		kf_params_.focal_x = intrinsic.fx;
		kf_params_.focal_y = intrinsic.fy;

        rec.set_intrinsic(
                intrinsic.fx, intrinsic.fy,
                intrinsic.ppx, intrinsic.ppy);

		std::cout << kf_params_.image_width << std::endl;
		std::cout << kf_params_.image_height << std::endl;

		// TODO : consider rectification
		//std::memcpy(params.d, intrinsic.coeffs, sizeof(params.d));
	}

	void update_profile(bool force=false){
		if(force || profile_changed(
					pl_.get_active_profile().get_streams(),
					prof_.get_streams())){
			prof_ = pl_.get_active_profile();

			// update depth scale
			for(auto& s : prof_.get_device().query_sensors() ){
				if (rs2::depth_sensor dpt = s.as<rs2::depth_sensor>() ){
					depth_scale_ = dpt.get_depth_scale();
				}
			}

			align_to_ = find_stream_to_align(prof_.get_streams());
			align_.reset( new rs2::align(align_to_) );
		}
	}

	void get(){
		auto fs = pl_.wait_for_frames();
		update_profile();

		auto fs_proc = align_->process( fs );
		rs2::video_frame vf = fs_proc.first(align_to_);
		rs2::depth_frame df = fs_proc.get_depth_frame();


		set_color(vf, cmap);
		//set_depth(df, 1000.0f * depth_scale_, dmap); // << 1000.0f necessary for kinectfusion
		set_depth(df, depth_scale_, dmap);

        stamp_ = fs.get_timestamp();
	}
	void proc(){
        bool suc = this->rec.process_frame(dmap, cmap, stamp_);
		//bool suc = kf_pl_->process_frame(dmap, cmap);
		if(!suc){
			std::cout << "KinectFusion Unsuccessful!!" << std::endl;
		}
	}

	void show(){
        std::cout << "SHOW" << std::endl;
		//viz img
		cv::imshow("col", cmap);
		//viz disparity
		double mnv, mxv;
		cv::minMaxIdx(dmap, &mnv, &mxv);
		cv::imshow("dpt", dmap * (1.0 / mxv));

		//cv::imshow("Pipeline Output", kf_pl_->get_last_model_frame());
	}

	void save(){
		// Retrieve camera poses
		auto poses = kf_pl_->get_poses();

		std::ofstream f("/tmp/poses.txt");
		if (f.is_open()){
			for(auto& p : poses){
				f << p << std::endl;
			}
		}

		// Export surface mesh
		auto mesh = kf_pl_->extract_mesh();
		kinectfusion::export_ply("/tmp/mesh.ply", mesh);

		// Export pointcloud
		auto pointcloud = kf_pl_->extract_pointcloud();
		kinectfusion::export_ply("/tmp/pointcloud.ply", pointcloud);
	}

};

int main() {

	RS2 rs2;

	for(int i=0; i<10; ++i){
		// warm-up
		rs2.get();
		cv::waitKey( 100 );
        //rs2.show();
	}

    //cv::waitKey( 0 );

	while(true){
		rs2.get();
		rs2.proc();
		rs2.show();

		int k = cv::waitKey( 1 );
		if(k == 27){
		   	break;
		}
	}
	rs2.save();
}
//catch (const rs2::error & e)
//{
//    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
//    return EXIT_FAILURE;
//}
//catch (const std::exception& e){
//    std::cerr << "HMM?" << e.what() << std::endl;
//    return EXIT_FAILURE;
//}
