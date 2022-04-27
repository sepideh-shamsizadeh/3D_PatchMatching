#include <pm.h>
#include <random>
#include <iostream>
#include <fstream>

#define WINDOW_SIZE 35
#define MAX_DISPARITY 60
#define DISP_PENALTY 120

using namespace std;

namespace pm {

  void compute_greyscale_gradient(const ::cv::Mat3b &frame, cv::Mat2f &grad) {
    int scale = 1, delta = 0;
    cv::Mat gray, x_grad, y_grad;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, x_grad, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(gray, y_grad, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
    x_grad = x_grad / 8.f;
    y_grad = y_grad / 8.f;

    for (int y = 0; y < frame.rows; ++y) {
      for (int x = 0; x < frame.cols; ++x) {
        grad(y, x)[0] = x_grad.at<float>(y, x);
        grad(y, x)[1] = y_grad.at<float>(y, x);
      }
    }
  }


  PatchMatch::PatchMatch(float alpha, float gamma, float tau_c, float tau_g)
          : alpha_(alpha), gamma_(gamma), tau_c_(tau_c), tau_g_(tau_g) {}


  float PatchMatch::dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq, const cv::Vec2f &pg, const cv::Vec2f &qg) {
    float cost_c = cv::norm(pp - qq, cv::NORM_L1);
    float cost_g = cv::norm(pg - qg, cv::NORM_L1);
    cost_c = std::min(cost_c, tau_c_);
    cost_g = std::min(cost_g, tau_g_);
    return (1 - alpha_) * cost_c + alpha_ * cost_g;
  }

  void PatchMatch::precomputeCostVolume(int ws, int cpv)
  {
    volumes_[cpv].resize(MAX_DISPARITY+1);
    for (int d=0; d<=MAX_DISPARITY; ++d)
    {
      std::cout<<"prepr d: "<<d<<std::endl;
      volumes_[cpv][d]=cv::Mat1f(rows_, cols_);
#pragma omp parallel for
      for (int y = 0; y < rows_; ++y)
      {
        for (int x = 0; x < cols_; ++x)
        {
          float cost=disp_match_cost((float) d, x,y,ws,cpv);
          volumes_[cpv][d].at<float>(y,x)=cost;
        }
      }
    }

    std::stringstream name; name<<"cost_volume"<<cpv<<".bin";
    std::ofstream f_vol(name.str(), ios::out | ios::binary ); 
    f_vol<<rows_<<endl<<cols_<<endl<<MAX_DISPARITY<<endl;
    for (int d=0; d<=MAX_DISPARITY; ++d)
      for (int y = 0; y < rows_; ++y)
        for (int x = 0; x < cols_; ++x)
        {
          float cost=volumes_[cpv][d].at<float>(y,x);
          f_vol<<cost<<endl;
        }
    f_vol.close();
  }

  void PatchMatch::loadPrecomputedCostVolumes(std::string dir)
  {
    for(int i=0; i<2; ++i)
    {
      std::stringstream name; name<<dir<<"/cost_volume"<<i<<".bin";
      std::ifstream f_vol(name.str(), ios::in | ios::binary); 
      int rows; f_vol>>rows;
      int cols; f_vol>>cols;
      int max_d; f_vol>>max_d;
  
      volumes_[i].resize(max_d+1);
      for (int d=0; d<=max_d; ++d)
      {
        volumes_[i][d]=cv::Mat1f(rows, cols);
        for (int y = 0; y < rows; ++y)
          for (int x = 0; x < cols; ++x)
          {
            float cost; f_vol>>cost;
            volumes_[i][d].at<float>(y,x)=cost;
            
          }
      }
      f_vol.close();
    }
  }


  // aggregated matchig cost for a pixel
  float PatchMatch::disp_match_cost(float dsp, int cx, int cy, int ws, int cpv) {
    int sign = -1 + 2 * cpv;

    float cost = 0;
    int half = ws / 2;

    const cv::Mat3b &f1 = views_[cpv];
    const cv::Mat3b &f2 = views_[1 - cpv];
    const cv::Mat2f &g1 = grads_[cpv];
    const cv::Mat2f &g2 = grads_[1 - cpv];
    const cv::Mat &w1 = weigs_[cpv];
    
    for (int x = cx - half; x <= cx + half; ++x) {
      for (int y = cy - half; y <= cy + half; ++y) {
        if (!inside(x, y, 0, 0, f1.cols, f1.rows))
          continue;

        if (dsp < 0 || dsp > MAX_DISPARITY) {
          cost += DISP_PENALTY;
        } else {
          // find matching point in other view
          float match = x + sign * dsp;
          int x_match = (int) match;

          float wm = 1 - (match - x_match);

          if (x_match > f1.cols - 2)
            x_match = f1.cols - 2;
          if (x_match < 0)
            x_match = 0;

          // and evaluating its color and gradinet (averaged)
          cv::Vec3b mcolo = vecAverage(f2(y, x_match), f2(y, x_match + 1), wm);
          cv::Vec2b mgrad = vecAverage(g2(y, x_match), g2(y, x_match + 1), wm);

          float w = w1.at<float>(cv::Vec<int, 4>{cy, cx, y - cy + half, x - cx + half});
          cost += w * dissimilarity(f1(y, x), mcolo, g1(y, x), mgrad);
        }
      }
    }

    return cost;
  }

  // aggregated precomputed matchig cost for a pixel
  float PatchMatch::precomputed_disp_match_cost(float dsp, int cx, int cy, int cpv) {
    if (dsp < 0 || dsp > MAX_DISPARITY) 
      return 200000;
    float ret=volumes_[cpv][(int)dsp].at<float>(cy,cx);
    return ret;
  }

  void PatchMatch::precompute_pixels_weights(const cv::Mat3b &frame, cv::Mat &weights, int ws) {
    int half = ws / 2;

#pragma omp parallel for
    for (int cx = 0; cx < frame.cols; ++cx)
      for (int cy = 0; cy < frame.rows; ++cy)

        for (int x = cx - half; x <= cx + half; ++x)
          for (int y = cy - half; y <= cy + half; ++y)
            if (inside(x, y, 0, 0, frame.cols, frame.rows))
              weights.at<float>(cv::Vec<int, 4>{cy, cx, y - cy + half, x - cx + half}) = weight(frame(cy, cx),
                                                                                                frame(y, x),
                                                                                                gamma_);
  }


  void PatchMatch::initialize_random_disps(cv::Mat1f &disps, float max_d) {
    cv::RNG random_generator;
    const int RAND_HALF = RAND_MAX / 2;

    for (int y = 0; y < rows_; ++y) {
      for (int x = 0; x < cols_; ++x) {
        disps.at<float>(y, x) = random_generator.uniform(.0f, max_d); // random disparity

      }
    }
  }


  void PatchMatch::evaluate_disps_cost(int cpv) {
#pragma omp parallel for
    for (int y = 0; y < rows_; ++y)
      for (int x = 0; x < cols_; ++x)
        costs_[cpv](y, x) = disp_match_cost(disps_[cpv].at<float>(y, x), x, y, WINDOW_SIZE, cpv);
  }


  // search for better disparity in the neighbourhood of a pixel
  // if iter is even then the function check the left and upper neighbours
  // if iter is odd then the function check the right and lower neighbours
  void PatchMatch::spatial_propagation(int x, int y, int cpv, int iter) {
    //INPUT:
    //x --> x coordinate
    //y --> y coordinate
    //cpv --> flag for selecting data related to left or right view
    //iter --> if iter is an odd number check for left or down else for right or up

    //CLASS VARIABLE TO USE:
    //disps_-->disparities, to access use disps_[cpv].at<float>(y, x)
    //costs_-->costs, to access use disps_[cpv].at<float>(y, x)


    //to compute costs use precomputed_disp_match_cost

    //to check if x and y values are valid use the function:
    // bool inside(int x, int y, int lbx, int lby, int ubx, int uby)
    //lbx=0, lby=0, ubx=cols_, uby=rows_

  }


  void PatchMatch::view_propagation(int x, int y, int cpv) {

    //INPUT:
    //x --> x coordinate
    //y --> y coordinate
    //cpv --> flag for selecting data related to left or right view


    //CLASS VARIABLE TO USE:
    //disps_-->disparities, to access use disps_[cpv].at<float>(y, x)
    //costs_-->costs, to access use disps_[cpv].at<float>(y, x)

    //Given the inputs the function should perturb the disparity at position (x,y) by a factor of delta_z
    //(i.e. new disparity(x,y) = old disparity(x,y) + delta_z), use max_delta_z and end_dz parameters to cycle between
    // different delta_z

    //to compute costs use precomputed_disp_match_cost

    //to check if x and y values are valid use the function:
    // bool inside(int x, int y, int lbx, int lby, int ubx, int uby)
    //lbx=0, lby=0, ubx=cols_, uby=rows_

  }


  void PatchMatch::disp_perturbation(int x, int y, int cpv, float max_delta_z, float end_dz) {

    //INPUT:
    //x --> x coordinate
    //y --> y coordinate
    //cpv --> flag for selecting data related to left or right view
    //max_delta_z --> max delta disparity
    //end_dz --> min delta disparity

    //CLASS VARIABLE TO USE:
    //disps-->disparities, to access use disps_[cpv].at<float>(y, x)
    //costs-->costs, to access use disps_[cpv].at<float>(y, x)

    //Given the inputs the function should perturb the disparity at position (x,y) by a factor of delta_z
    //(i.e. new disparity(x,y) = old disparity(x,y) + delta_z), use max_delta_z and end_dz parameters to cycle between
    // different delta_z (free to choose how to do)

    //to compute costs use precomputed_disp_match_cost

  }


  void PatchMatch::process_pixel(int x, int y, int cpv, int iter) 
  {
    // spatial propagation
    spatial_propagation(x, y, cpv, iter);

    // disparity refinement
    disp_perturbation(x, y, cpv, MAX_DISPARITY / 2, .5f);

    // view propagation
    view_propagation(x, y, cpv);
  }



  void PatchMatch::weighted_median_filter(int cx, int cy, cv::Mat1f &disparity, const cv::Mat &weights,
                                          const cv::Mat1b &valid, int ws, bool use_invalid) {
    int half = ws / 2;
    float w_tot = 0;
    float w = 0;

    std::vector<std::pair<float, float>> disps_w;

    for (int x = cx - half; x <= cx + half; ++x)
      for (int y = cy - half; y <= cy + half; ++y)
        if (inside(x, y, 0, 0, cols_, rows_) && (use_invalid || valid(y, x))) {
          cv::Vec<int, 4> w_ids({cy, cx, y - cy + half, x - cx + half});

          w_tot += weights.at<float>(w_ids);
          disps_w.push_back(std::make_pair(weights.at<float>(w_ids), disparity(y, x)));
        }

    std::sort(disps_w.begin(), disps_w.end());

    float med_w = w_tot / 2.0f;

    for (auto dw = disps_w.begin(); dw < disps_w.end(); ++dw) {
      w += dw->first;

      if (w >= med_w) {
        if (dw == disps_w.begin()) {
          disparity(cy, cx) = dw->second;
        } else {
          disparity(cy, cx) = ((dw - 1)->second + dw->second) / 2.0f;
        }
        //disparity(cy, cx) = dw->second;
      }
    }
  }


  void PatchMatch::set(const std::string dir)
  {
    // Reading images
	  cv::Mat3b img1 = cv::imread(dir+"/view0.png", cv::IMREAD_COLOR);
	  cv::Mat3b img2 = cv::imread(dir+"/view1.png", cv::IMREAD_COLOR);

    cv::Mat1b gt1 = cv::imread(dir+"/GT0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat1b gt2 = cv::imread(dir+"/GT1.png", cv::IMREAD_GRAYSCALE);


    std::cerr << "Loading precomputed cost volumes...\n";
    loadPrecomputedCostVolumes(dir);

    set(img1, img2, gt1, gt2);
  }

  void PatchMatch::set(const cv::Mat3b &img1, const cv::Mat3b &img2,  const cv::Mat1f &gt1, const cv::Mat1f &gt2) {
    views_[0] = img1;
    views_[1] = img2;

    gt_[0] = gt1;
    gt_[1] = gt2;

    rows_ = img1.rows;
    cols_ = img1.cols;

    // pixels neighbours weights
    std::cerr << "Precomputing pixels weight...\n";
    int wmat_sizes[] = {rows_, cols_, WINDOW_SIZE, WINDOW_SIZE};
    this->weigs_[0] = cv::Mat(4, wmat_sizes, CV_32F);
    this->weigs_[1] = cv::Mat(4, wmat_sizes, CV_32F);
    precompute_pixels_weights(img1, weigs_[0], WINDOW_SIZE);
    precompute_pixels_weights(img2, weigs_[1], WINDOW_SIZE);

    // greyscale images gradient
    std::cerr << "Evaluating images gradient...\n";
    grads_[0] = cv::Mat2f(rows_, cols_);
    grads_[1] = cv::Mat2f(rows_, cols_);
    compute_greyscale_gradient(img1, grads_[0]);
    compute_greyscale_gradient(img2, grads_[1]);

    // pixels' disparities random inizialization
    std::cerr << "Precomputing random disparities...\n";
    disps_[0] = cv::Mat1f(rows_, cols_);
    disps_[1] = cv::Mat1f(rows_, cols_);

    initialize_random_disps(disps_[0], MAX_DISPARITY);
    initialize_random_disps(disps_[1], MAX_DISPARITY);

    // initial disparity costs evaluation
    std::cerr << "Evaluating initial disparity cost...\n";
    costs_[0] = cv::Mat1f(rows_, cols_);
    costs_[1] = cv::Mat1f(rows_, cols_);
    evaluate_disps_cost(0);
    evaluate_disps_cost(1);

    //std::cerr << "Precompute cost volume left...\n";
    //precomputeCostVolume(WINDOW_SIZE, 0);
    //std::cerr << "Precompute cost volume right...\n";
    //precomputeCostVolume(WINDOW_SIZE, 1);

  }


  void PatchMatch::process(int iterations) 
  {
    std::cerr << "Processing left and right views...\n";
    for (int iter = 0 ; iter < iterations; ++iter) 
    {
      bool iter_type = (iter % 2 == 0);
      std::cerr << "Iteration " << iter + 1 << "/" << iterations << "\r";

      // PROCESS LEFT AND RIGHT VIEW IN SEQUENCE (0=left view, 1=right view)
      for (int work_view = 0; work_view < 2; ++work_view) 
      {
        if (iter_type) { // FORWARD SCANNING
          for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x)
              process_pixel(x, y, work_view, iter);
        } else { // BACKWARD SCANNING
          for (int y = rows_ - 1; y >= 0; --y)
            for (int x = cols_ - 1; x >= 0; --x)
              process_pixel(x, y, work_view, iter);
        }
      }
    }
    std::cerr << std::endl;
  }

  void PatchMatch::fill_invalid_pixels(int y, int x, int cpv, const cv::Mat1b &validity)
  {
    int x_lft = x - 1;
    int x_rgt = x + 1;

    while(!validity(y, x_lft) && x_lft >= 0)
      --x_lft;

    while(!validity(y, x_rgt) && x_lft < cols_)
      ++x_rgt;

    int best_x = x;

    if(x_lft >= 0 && x_rgt < cols_)
    {
      float disp_l = disps_[cpv].at<float>(y, x_lft);
      float disp_r = disps_[cpv].at<float>(y, x_rgt);
      best_x = (disp_l < disp_r) ? x_lft : x_rgt;
    }
    else if(x_lft >= 0)
      best_x = x_lft;
    else if(x_rgt < cols_)
      best_x = x_rgt;

    disps_[cpv].at<float>(y, x) = disps_[cpv].at<float>(y, best_x);
  }


  void PatchMatch::postProcess()
  {
    std::cerr<<"Executing post-processing...\n";

    // checking pixels disparity validity
    cv::Mat1b lft_validity(rows_, cols_, (unsigned char)false);
    cv::Mat1b rgt_validity(rows_, cols_, (unsigned char)false);

    for(int y=0; y < rows_; ++y)
    {
      for(int x=0; x < cols_; ++x)
      {
        int x_rgt_match = std::max(0.f, std::min((float)cols_, x - disps_[0](y, x)));
        lft_validity(y, x) = (std::abs(disps_[0](y, x) - disps_[1](y, x_rgt_match)) <= 1);

        int x_lft_match = std::max(0.f, std::min((float)rows_, x + disps_[1](y, x)));
        rgt_validity(y, x) = (std::abs(disps_[1](y, x) - disps_[0](y, x_lft_match)) <= 1);
      }
    }


    // fill-in holes related to invalid pixels
    for(int y=0; y < rows_; y++)
    {
      for (int x=0; x < cols_; x++)
      {
        if (!lft_validity(y, x))
          fill_invalid_pixels(y, x, 0, lft_validity);

        if (!rgt_validity(y, x))
          fill_invalid_pixels(y, x, 1, rgt_validity);
      }
    }


    // applying weighted median filter to left and right view respectively
    for(int x=0; x<cols_; ++x)
    {
      for(int y=0; y<rows_; ++y)
      {
        weighted_median_filter(x, y, disps_[0], weigs_[0], lft_validity, WINDOW_SIZE, false);
        weighted_median_filter(x, y, disps_[1], weigs_[1], rgt_validity, WINDOW_SIZE, false);
      }
    }
  }

  float PatchMatch::computeMSE(int cpv)
  {
    if(gt_[cpv].empty())
      return -1;
    cv::Mat1f container[2];
    cv::normalize(gt_[cpv], container[0], 0, 63.75, cv::NORM_MINMAX);
    container[1] = disps_[cpv].clone();
    cv::Mat1f  mask = min(gt_[cpv],1);
    cv::multiply(container[1], mask, container[1], 1);
    float error = 0;
    for (int y=0; y<rows_; ++y)
    {
      for (int x=0; x<cols_; ++x)
      {
        float diff = container[0](y,x) - container[1](y,x);
        error+=(diff*diff);
      }
    }
    error = error/(rows_*cols_);
    //error = cv::quality::QualityMSE::compute(container[0], container[1], cv::noArray())[0];
    return error;
  }
}
