#ifndef PATCH_MATCH_H
#define PATCH_MATCH_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <exception>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>


namespace pm
{
	class PatchMatch
	{
		public:
			PatchMatch(float alpha, float gamma, float tau_c, float tau_g);
			
			PatchMatch(const PatchMatch &pm) = delete;
			
			PatchMatch& operator=(const PatchMatch &pm) = delete;

      /**
       * @brief Function for reading images and precomputed cost volumes from the dataset directory.
       *
       * @param[in] dir path to the input dataset.
       *
       * @return void
       *
       * The dataset directory should contain the left/right images and left/right precomputed cost volumes (view0.png/view1.png and cost_volume0.bin/cost_volume1.bin).
       *
       * */
      void set(const std::string dir);

			void set(const cv::Mat3b &img1, const cv::Mat3b &img2, const cv::Mat1f &gt1, const cv::Mat1f &gt2);
			
      /**
       * @brief Main loop function for forward and backward scanning of left and right views.
       *
       * @param[in] iterations number of iterations for forward and backward scanning. (typically 3 iterations are enough)
       *
       * @return void
       *
       * For each scanned pixel, the function 'process_pixel(...)' is called for disparity propagation and random search.
       *
       * */
			void process(int iterations);
			
      /**
       * @brief Post processing for output disparity maps refinement.
       *
       * @return void
       *
       * The post processing includes left/right consistensy check and disparity maps filtering and smoothing.
       *
       * */
			void postProcess();

      void fill_invalid_pixels(int y, int x, int cpv, const cv::Mat1b &validity);

			cv::Mat1f getLeftDisparityMap() const;
			
			cv::Mat1f getRightDisparityMap() const;

      float computeMSE(int cpv);

			float alpha_;
			float gamma_;
			float tau_c_;
			float tau_g_;
		
		private:

      void precomputeCostVolume(int ws, int cpv);

      /**
       * @brief Load precomputed cost volumes from the data directory.
       *
       * @param[in] dir path to the dataset directory.
       *
       * @return void
       *
       * It's assumed that the precomputed cost volumes exist with names cost_volume0.bin and cost_volume1.bin (for left and right views respectively).
       *
       * */
      void loadPrecomputedCostVolumes(std::string dir);
		
			float dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq, const cv::Vec2f &pg, const cv::Vec2f &qg);
	
      /**
       * @brief Get the precomputed matching cost of a patch centered at (cx, cy) in the current view with the patch centered at (cx+disp, cy) in the other view.
       *
       * @param[in] disp input disparity.
       * @param[in] cx x-coordinate of the center of the patch.
       * @param[in] cy y-coordinate of the center of the patch.
       * @param[in] cpv flag defining the current view (0=left, 1=right).
       *
       * @return matching cost.
       *
       * */
			float precomputed_disp_match_cost(float disp, int cx, int cy, int cpv);

      /**
       * @brief compute the matching cost of a patch centered at (cx, cy) in the current view with the patch centered at (cx+disp, cy) in the other view.
       *
       * @param[in] disp input disparity.
       * @param[in] cx x-coordinate of the center of the patch.
       * @param[in] cy y-coordinate of the center of the patch.
       * @param[in] ws patch size.
       * @param[in] cpv flag defining the current view (0=left, 1=right).
       *
       * @return matching cost.
       *
       * */
      float disp_match_cost(float disp, int cx, int cy, int ws, int cpv);
			
			void precompute_pixels_weights(const cv::Mat3b &frame, cv::Mat &weights, int ws);
			
      /**
       * @brief Random initialization of the disparity map.
       *
       * @param[out] disps output disparity map.
       * @param[in] max_d maximum disparity value.
       *
       * @return void.
       *
       * The disparity map is filled with random values in the range [0,max_d].
       *
       * */
			void initialize_random_disps(cv::Mat1f &disps, float max_d);
			
			void evaluate_disps_cost(int cpv);
			
			void spatial_propagation(int x, int y, int cpv, int iter);
			
			void view_propagation(int x, int y, int cpv);
			
			void disp_perturbation(int x, int y, int cpv, float max_delta_z, float end_dz);
			
      /**
       * @brief Core function that implements the pixel's disparity random search and propagation.
       *
       * @param[in] x x-coordinate of the current pixel.
       * @param[in] y y-coordinate of the current pixel.
       * @param[in] cpv flag defining the current view (0=left, 1=right).
       * @param[in] iter current loop iteration.
       *
       * @return void.
       *
       * */
			void process_pixel(int x, int y, int cpv, int iter);

			void weighted_median_filter(int cx, int cy, cv::Mat1f &disparity, const cv::Mat &weights, const cv::Mat1b &valid, int ws, bool use_invalid);
			
			
			cv::Mat3b views_[2];			          // left and right view images
      cv::Mat1f gt_[2];			              // left and right gt disparities
			cv::Mat2f grads_[2];			          // pixels greyscale gradient for both views
			cv::Mat1f disps_[2];			          // left and right disparity maps
			cv::Mat1f costs_[2];			          // left and right costs
			cv::Mat weigs_[2];			            // precomputed pixels window weights
      std::vector<cv::Mat1f> volumes_[2]; // left and right precomputed cost volumes
			int rows_;
			int cols_;
		
	};
	
	
	
	inline cv::Mat1f PatchMatch::getLeftDisparityMap() const
	{
		return this->disps_[0].clone();
	}
	
	
	inline cv::Mat1f PatchMatch::getRightDisparityMap() const
	{
		return this->disps_[1].clone();
	}
	
	
	// consider preallocated gradients matrix
	void compute_greyscale_gradient(const::cv::Mat3b &frame, cv::Mat2f &gradient);		
	
	
	inline bool inside(int x, int y, int lbx, int lby, int ubx, int uby)
	{
		return lbx <= x && x < ubx && lby <= y && y < uby;
	}


	inline float weight(const cv::Vec3f &p, const cv::Vec3f &q, float gamma=10.0f)
	{
		return std::exp(-cv::norm(p-q, cv::NORM_L1) / gamma);
	}
	
	
	template <typename T>
	inline T vecAverage(const T &x, const T &y, float wx)
	{
		return wx * x + (1 - wx) * y;
	}
}

#endif
