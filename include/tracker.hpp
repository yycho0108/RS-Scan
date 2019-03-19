#ifndef __TRACK_HPP__
#define __TRACK_HPP__

#include "types.hpp"
#include "matcher.hpp"
#include <opencv2/features2d.hpp>

class Tracker{
	// processing handles
    cv::Ptr<cv::SparsePyrLKOpticalFlow> tracker_;

    // camera parameters
	cv::Mat K_, D_;

	// data cache
	FrameData prv_;

  public:
	  Tracker();
	  ~Tracker();
	  //void track(
      //        const FrameData& kf0,
      //        const FrameData& kf1,
      //        std::vector<cv::DMatch>& m01
      //        );

	  void track(
              const FrameData& kf0,
              const FrameData& kf1,
              std::vector<size_t>& i0,
              std::vector<size_t>& i1,
              );
};
#endif
