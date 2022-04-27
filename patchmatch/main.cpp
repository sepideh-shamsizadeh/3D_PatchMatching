#include <iostream>
#include <opencv2/opencv.hpp>
#include <pm.h>

int main(int argc, char** argv)
{
	const float alpha =  0.9f;
	const float gamma = 10.0f;
	const float tau_c = 10.0f;
	const float tau_g =  2.0f;
	
	pm::PatchMatch patch_match(alpha, gamma, tau_c, tau_g);

  // Set the data directory and initialize algorithm
	patch_match.set(std::string(argv[1]));

  // Run the algorithm
	patch_match.process(3);

  // Refine estimated disparity maps
  patch_match.postProcess();
	

  // Get the output disparity maps
	cv::Mat1f disp1 = patch_match.getLeftDisparityMap();
	cv::Mat1f disp2 = patch_match.getRightDisparityMap();

  // Compute Mean Squared Error (if ground truth exists)
  float error0 = patch_match.computeMSE(0);
  float error1 = patch_match.computeMSE(1);
  std::cerr<<"Left Image MSE error: "<<error0<<std::endl;
  std::cerr<<"Right Image MSE error: "<<error1<<std::endl;

  // Save results on disk
	try
	{
    cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);
	  cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);
		cv::imwrite("left_disparity.png", disp1);
		cv::imwrite("right_disparity.png", disp2);
	} 
	catch(std::exception &e)
	{
		std::cerr << "Disparity save error.\n" <<e.what();
		return 1;
	}

  // Show results
  cv::Mat show1, show2;
  disp1.convertTo(show1, CV_8U);
  disp2.convertTo(show2, CV_8U);
  cv::imshow("left disparity map", show1);
  cv::imshow("right disparity map", show2);
  cv::waitKey(0);

	return 0;
}
