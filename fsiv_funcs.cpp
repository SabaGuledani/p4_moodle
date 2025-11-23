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
    // dilate edges to expand edge regions
    cv::Mat dilated_edges;
    if (dilate_radius > 0)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                    cv::Size(2 * dilate_radius + 1, 2 * dilate_radius + 1));
        cv::dilate(edges_u, dilated_edges, kernel);
    }
    else
    {
        dilated_edges = edges_u.clone();
    }
    
    // combine motion mask with dilated edges using union (OR operation)
    cv::bitwise_or(motion_u, dilated_edges, refined_u);
}

void fsiv_apply_background_blur(
    const cv::Mat& bgr, const cv::Mat& fg_mask_u, int blur_radius, cv::Mat& out_bgr)
{
    // if blur radius is 0, just copy the original image
    if (blur_radius == 0)
    {
        bgr.copyTo(out_bgr);
        return;
    }
    
    // apply gaussian blur to create blurred background
    cv::Mat blurred;
    int kernel_size = 2 * blur_radius + 1;
    cv::GaussianBlur(bgr, blurred, cv::Size(kernel_size, kernel_size), 0);
    
    // convert mask to float [0..1] for blending
    cv::Mat mask_f;
    fg_mask_u.convertTo(mask_f, CV_32F, 1.0 / 255.0);
    
    // create 3-channel mask for color image blending
    std::vector<cv::Mat> mask_channels;
    mask_channels.push_back(mask_f);
    mask_channels.push_back(mask_f);
    mask_channels.push_back(mask_f);
    cv::Mat mask_3ch;
    cv::merge(mask_channels, mask_3ch);
    
    // convert images to float for blending
    cv::Mat bgr_f, blurred_f;
    bgr.convertTo(bgr_f, CV_32F);
    blurred.convertTo(blurred_f, CV_32F);
    
    // composite: foreground from original, background from blurred
    // mask_3ch: 1.0 = foreground (keep original), 0.0 = background (apply blur)
    cv::Mat result_f;
    cv::multiply(bgr_f, mask_3ch, result_f);  // foreground part from original
    
    // create inverted mask for background: 1.0 - mask_3ch
    cv::Mat ones = cv::Mat::ones(mask_3ch.size(), mask_3ch.type());
    cv::Mat inv_mask_3ch;
    cv::subtract(ones, mask_3ch, inv_mask_3ch);
    
    cv::Mat bg_part;
    cv::multiply(blurred_f, inv_mask_3ch, bg_part);  // background part from blurred
    result_f = result_f + bg_part;
    
    // convert back to 8bit unsigned
    result_f.convertTo(out_bgr, CV_8U);
}
