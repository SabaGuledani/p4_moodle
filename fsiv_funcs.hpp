/**
 * @file fsiv_funcs.hpp
 * @brief Declarations for Smart Background Filtering (motion + edges).
 * @author FSIV
 */

#ifndef FSIV_FUNCS_HPP
#define FSIV_FUNCS_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief Clamp a value to [lo, hi] (portable; avoids std::clamp).
 */
template<typename T>
inline T fsiv_clamp(T v, T lo, T hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/**
 * @brief Convert BGR frame to single-channel grayscale (CV_8U).
 * @param bgr Input BGR frame.
 * @param gray Output grayscale (CV_8U).
 */
void fsiv_to_grayscale(const cv::Mat& bgr, cv::Mat& gray);

/**
 * @brief Compute dense optical flow (Farnebäck).
 * @param prev_gray Previous grayscale frame (CV_8U).
 * @param gray Current grayscale frame (CV_8U).
 * @param flow Output 2-channel flow (CV_32FC2).
 * @param pyr_scale Farnebäck param.
 * @param levels Farnebäck param.
 * @param winsize Farnebäck param.
 * @param iterations Farnebäck param.
 * @param poly_n Farnebäck param.
 * @param poly_sigma Farnebäck param.
 */
void fsiv_compute_optical_flow_farneback(
    const cv::Mat& prev_gray,
    const cv::Mat& gray,
    cv::Mat& flow,
    double pyr_scale=0.5, int levels=3, int winsize=15,
    int iterations=3, int poly_n=5, double poly_sigma=1.2);

/**
 * @brief Compute magnitude (pixels/frame) of a 2-channel flow.
 * @param flow Input flow (CV_32FC2).
 * @param mag Output magnitude (CV_32F).
 */
void fsiv_flow_magnitude(const cv::Mat& flow, cv::Mat& mag);

/**
 * @brief Threshold a float magnitude image into a binary mask (0/255).
 * @param mag Input magnitude (CV_32F).
 * @param t_flow Threshold in pixels/frame.
 * @param mask Output CV_8U mask (0 or 255).
 */
void fsiv_motion_mask_from_mag(const cv::Mat& mag, float t_flow, cv::Mat& mask);

/**
 * @brief Exponential running average (temporal smoothing) for binary masks.
 * @param prev_mask_f Previous state [0..1] (CV_32F). If empty, initialized.
 * @param curr_mask_u Binary input mask (CV_8U, {0,255}).
 * @param alpha Memory factor in [0,1]. Higher = slower adaptation.
 * @param out_mask_u Smoothed mask (CV_8U, {0,255}).
 */
void fsiv_update_running_mask(
    cv::Mat& prev_mask_f, const cv::Mat& curr_mask_u, float alpha, cv::Mat& out_mask_u);

/**
 * @brief Compute Canny edges on a grayscale image.
 * @param gray Input CV_8U grayscale.
 * @param low Lower hysteresis threshold.
 * @param high Upper hysteresis threshold.
 * @param edges Output binary CV_8U edges (0/255).
 */
void fsiv_compute_edges(const cv::Mat& gray, int low, int high, cv::Mat& edges);

/**
 * @brief Refine a motion mask using dilated edges (union and clean-up).
 * @param motion_u Binary CV_8U motion mask.
 * @param edges_u Binary CV_8U edges mask.
 * @param dilate_radius Radius (in pixels) to dilate edges (>=0).
 * @param refined_u Output refined binary CV_8U mask (0/255).
 */
void fsiv_refine_foreground_mask(
    const cv::Mat& motion_u, const cv::Mat& edges_u, int dilate_radius, cv::Mat& refined_u);

/**
 * @brief Apply background blur and keep foreground sharp.
 * @param bgr Input BGR frame (CV_8U, 3-ch).
 * @param fg_mask_u Binary CV_8U mask (255=foreground).
 * @param blur_radius Gaussian radius (kernel size = 2*r+1; if r==0 -> copy).
 * @param out_bgr Output composited frame.
 */
void fsiv_apply_background_blur(
    const cv::Mat& bgr, const cv::Mat& fg_mask_u, int blur_radius, cv::Mat& out_bgr);



#endif // FSIV_FUNCS_HPP
