/**
 * @file fsiv_funcs.cpp
 * @brief Definitions for Smart Background Filtering helpers (motion + edges).
 */

#include "fsiv_funcs.hpp"
#include <cmath>

void fsiv_to_grayscale(const cv::Mat& bgr, cv::Mat& gray)
{
    // convert bgr image to single channel grayscale
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
}

void fsiv_compute_optical_flow_farneback(
    const cv::Mat& prev_gray,
    const cv::Mat& gray,
    cv::Mat& flow,
    double pyr_scale, int levels, int winsize,
    int iterations, int poly_n, double poly_sigma)
{
    // compute dense optical flow using farneback algorithm
    // flow will be a 2-channel image with x and y displacement vectors
    cv::calcOpticalFlowFarneback(
        prev_gray, gray, flow,
        pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma,
        cv::OPTFLOW_FARNEBACK_GAUSSIAN
    );
}

void fsiv_flow_magnitude(const cv::Mat& flow, cv::Mat& mag)
{
    // split flow into x and y components
    std::vector<cv::Mat> flow_channels;
    cv::split(flow, flow_channels);
    
    // compute magnitude as sqrt(x^2 + y^2) for each pixel
    cv::magnitude(flow_channels[0], flow_channels[1], mag);
}

void fsiv_motion_mask_from_mag(const cv::Mat& mag, float t_flow, cv::Mat& mask)
{
    // threshold magnitude to detect motion areas
    // pixels with magnitude above threshold are marked as foreground - 255
    cv::threshold(mag, mask, t_flow, 255.0, cv::THRESH_BINARY);
    
    // convert to 8bit unsigned int
    mask.convertTo(mask, CV_8U);
}

void fsiv_update_running_mask(
    cv::Mat& prev_mask_f, const cv::Mat& curr_mask_u, float alpha, cv::Mat& out_mask_u)
{
    // initialize previous mask if empty (first frame)
    if (prev_mask_f.empty())
    {
        curr_mask_u.convertTo(prev_mask_f, CV_32F, 1.0 / 255.0);
    }
    
    // convert current mask to float [0..1] for computation
    cv::Mat curr_mask_f;
    curr_mask_u.convertTo(curr_mask_f, CV_32F, 1.0 / 255.0);
    
    // apply exponential running average: m = alpha*m_old + (1-alpha)*m_new
    cv::addWeighted(prev_mask_f, alpha, curr_mask_f, 1.0f - alpha, 0.0, prev_mask_f);
    
    // convert back to 8bit unsigned and update output
    prev_mask_f.convertTo(out_mask_u, CV_8U, 255.0);
}

void fsiv_compute_edges(const cv::Mat& gray, int low, int high, cv::Mat& edges)
{
    // detect edges using canny operator with hysteresis thresholds
    // low threshold for weak edges, high threshold for strong edges
    cv::Canny(gray, edges, low, high);
}

void fsiv_refine_foreground_mask(
    const cv::Mat& motion_u, const cv::Mat& edges_u, int dilate_radius, cv::Mat& refined_u)
{
    // TODO: Implement mask refinement using edges
}

void fsiv_apply_background_blur(
    const cv::Mat& bgr, const cv::Mat& fg_mask_u, int blur_radius, cv::Mat& out_bgr)
{
    // TODO: Implement background blur with foreground preservation
}
