#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>

#include <algorithm>
#include <memory>
#include <cstring>

int main(){
	// basic fetch parameters
	int w = 640, h = 480, fps = 30;

	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_INFRARED, 1, w, h, RS2_FORMAT_Y8, fps);
	cfg.enable_stream(RS2_STREAM_INFRARED, 2, w, h, RS2_FORMAT_Y8, fps);
	rs2::pipeline pl;
	rs2::pipeline_profile prof = pl.start(cfg);

	while(1){
		rs2::frameset fs = pl.wait_for_frames();

		// 1-based index? what?
		rs2::video_frame irf_l = fs.get_infrared_frame( 1 );
		rs2::video_frame irf_r = fs.get_infrared_frame( 2 );

		cv::Mat img_l = cv::Mat(cv::Size(w,h), CV_8UC1, (void*)irf_l.get_data());
		cv::Mat img_r = cv::Mat(cv::Size(w,h), CV_8UC1, (void*)irf_r.get_data());

		cv::imshow("img_l", img_l);
		cv::imshow("img_r", img_r);

		char k = cv::waitKey( 1 );
		if(k == 27 || k == 'q'){
			break;
		}
	}
}
