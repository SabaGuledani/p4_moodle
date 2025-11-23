/**
 * @file fsiv_main.cpp
 * @brief Smart Background Filtering (motion + edges) demo with GUI & CLI.
 * @author FSIV
 *
 *
 * Run examples:
 *   ./fsiv_smartfilter --camera=0
 *   ./fsiv_smartfilter --video=sample.mp4 --alpha=0.8 --tflow=0.6 --blur=7 --canny_low=50 --canny_high=150
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "fsiv_funcs.hpp"

static const char* KEYS =
    "{help h ? |      | print this message }"
    "{video    |      | path to input video file }"
    "{camera   | -1   | camera index (>=0) }"    
    "{alpha    | 0.80 | running-average alpha in [0,1] }"
    "{tflow    | 0.50 | motion threshold (pixels/frame) }"
    "{blur     | 7    | Gaussian blur radius (0=no blur) }"
    "{canny_low| 50   | Canny low threshold [0..255] }"
    "{canny_high|150  | Canny high threshold [0..255] }"
    "{edge_dil | 2    | edge dilation radius (pixels) }"    
    "{out      |      | output video path }";

// trackbar callback functions (need to be global or static)
static int g_alpha_x100 = 80;
static int g_tflow_x10 = 5;
static int g_blur_radius = 7;
static int g_canny_low = 50;
static int g_canny_high = 150;
static int g_edge_dil = 2;

static void on_alpha_changed(int, void*) {}
static void on_tflow_changed(int, void*) {}
static void on_blur_changed(int, void*) {}
static void on_canny_low_changed(int, void*) {}
static void on_canny_high_changed(int, void*) {}
static void on_edge_dil_changed(int, void*) {}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, KEYS);
    parser.about("FSIV Smart Background Filter (motion + edges)");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // parse command line arguments
    std::string video_path = parser.get<std::string>("video");
    int camera_idx = parser.get<int>("camera");
    float alpha = parser.get<float>("alpha");
    float tflow = parser.get<float>("tflow");
    int blur_radius = parser.get<int>("blur");
    int canny_low = parser.get<int>("canny_low");
    int canny_high = parser.get<int>("canny_high");
    int edge_dil = parser.get<int>("edge_dil");
    std::string out_path = parser.get<std::string>("out");

    // initialize trackbar values from command line
    g_alpha_x100 = static_cast<int>(alpha * 100);
    g_tflow_x10 = static_cast<int>(tflow * 10);
    g_blur_radius = blur_radius;
    g_canny_low = canny_low;
    g_canny_high = canny_high;
    g_edge_dil = edge_dil;

    // open video capture
    cv::VideoCapture cap;
    if (!video_path.empty())
    {
        cap.open(video_path);
    }
    else if (camera_idx >= 0)
    {
        cap.open(camera_idx);
    }
    else
    {
        std::cerr << "Error: no video source specified (use --video or --camera)" << std::endl;
        return -1;
    }

    if (!cap.isOpened())
    {
        std::cerr << "Error: could not open video source" << std::endl;
        return -1;
    }

    // get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; // default fps if not available

    // setup output video writer
    cv::VideoWriter writer;
    if (!out_path.empty())
    {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(out_path, fourcc, fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened())
        {
            std::cerr << "Warning: could not open output video file" << std::endl;
        }
    }

    // GUI setup
    const std::string kWinOut = "FSIV Output";
    const std::string kWinDbg = "FSIV Debug (flow|edges|mask)";

    cv::namedWindow(kWinOut, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(kWinDbg, cv::WINDOW_AUTOSIZE);

    // create trackbars
    cv::createTrackbar("alpha x100", kWinOut, &g_alpha_x100, 100, on_alpha_changed);
    cv::createTrackbar("t_flow x0.1", kWinOut, &g_tflow_x10, 50, on_tflow_changed);
    cv::createTrackbar("blur radius", kWinOut, &g_blur_radius, 20, on_blur_changed);
    cv::createTrackbar("canny low", kWinOut, &g_canny_low, 255, on_canny_low_changed);
    cv::createTrackbar("canny high", kWinOut, &g_canny_high, 255, on_canny_high_changed);
    cv::createTrackbar("edge dil", kWinOut, &g_edge_dil, 10, on_edge_dil_changed);

    // processing variables
    cv::Mat frame, prev_gray, gray, flow, mag, motion_mask, prev_mask_f;
    cv::Mat edges, refined_mask, output;
    bool first_frame = true;
    int frame_count = 0;

    // main processing loop
    while (true)
    {
        if (!cap.read(frame) || frame.empty())
            break;

        // convert to grayscale
        fsiv_to_grayscale(frame, gray);

        if (!first_frame)
        {
            // compute optical flow
            fsiv_compute_optical_flow_farneback(prev_gray, gray, flow);

            // compute flow magnitude
            fsiv_flow_magnitude(flow, mag);

            // create motion mask from magnitude
            float current_tflow = g_tflow_x10 / 10.0f;
            fsiv_motion_mask_from_mag(mag, current_tflow, motion_mask);

            // apply temporal smoothing
            float current_alpha = g_alpha_x100 / 100.0f;
            fsiv_update_running_mask(prev_mask_f, motion_mask, current_alpha, motion_mask);

            // compute edges
            fsiv_compute_edges(gray, g_canny_low, g_canny_high, edges);

            // refine foreground mask using edges
            fsiv_refine_foreground_mask(motion_mask, edges, g_edge_dil, refined_mask);

            // apply background blur
            fsiv_apply_background_blur(frame, refined_mask, g_blur_radius, output);

            // create debug visualization according to Figure 2: 
            // a) Original, b) Blurred (11x11 box), c) Flow magnitude, d) Mask, e) Result
            cv::Mat blurred_box;
            cv::boxFilter(frame, blurred_box, -1, cv::Size(11, 11));

            // normalize flow magnitude for visualization and resize to match frame
            cv::Mat mag_vis;
            double mag_min, mag_max;
            cv::minMaxLoc(mag, &mag_min, &mag_max);
            if (mag_max > 0)
            {
                mag.convertTo(mag_vis, CV_8U, 255.0 / mag_max);
            }
            else
            {
                mag_vis = cv::Mat::zeros(mag.size(), CV_8U);
            }
            // resize to match frame size if needed
            if (mag_vis.size() != frame.size())
            {
                cv::resize(mag_vis, mag_vis, frame.size());
            }
            cv::cvtColor(mag_vis, mag_vis, cv::COLOR_GRAY2BGR);

            // create mask visualization (motion mask with dilation radius 5 as in Figure 2)
            cv::Mat mask_vis = motion_mask.clone();
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                      cv::Size(2 * 5 + 1, 2 * 5 + 1));
            cv::dilate(mask_vis, mask_vis, kernel);
            // resize to match frame size if needed
            if (mask_vis.size() != frame.size())
            {
                cv::resize(mask_vis, mask_vis, frame.size());
            }
            cv::cvtColor(mask_vis, mask_vis, cv::COLOR_GRAY2BGR);

            // ensure output is same size as frame
            cv::Mat output_resized = output;
            if (output.size() != frame.size())
            {
                cv::resize(output, output_resized, frame.size());
            }

            // ensure all images have exactly the same size and type before concatenation
            cv::Size target_size = frame.size();
            cv::Mat frame_vis, blurred_vis, mag_vis_resized, mask_vis_resized, output_vis;
            
            // ensure frame is CV_8UC3
            if (frame.type() != CV_8UC3)
                frame.convertTo(frame_vis, CV_8UC3);
            else
                frame.copyTo(frame_vis);
            if (frame_vis.size() != target_size)
                cv::resize(frame_vis, frame_vis, target_size);
            
            // ensure blurred_box is CV_8UC3 and correct size
            if (blurred_box.type() != CV_8UC3)
                blurred_box.convertTo(blurred_vis, CV_8UC3);
            else
                blurred_box.copyTo(blurred_vis);
            if (blurred_vis.size() != target_size)
                cv::resize(blurred_vis, blurred_vis, target_size);
                
            // ensure mag_vis is CV_8UC3 and correct size
            if (mag_vis.type() != CV_8UC3)
                mag_vis.convertTo(mag_vis_resized, CV_8UC3);
            else
                mag_vis.copyTo(mag_vis_resized);
            if (mag_vis_resized.size() != target_size)
                cv::resize(mag_vis_resized, mag_vis_resized, target_size);
                
            // ensure mask_vis is CV_8UC3 and correct size
            if (mask_vis.type() != CV_8UC3)
                mask_vis.convertTo(mask_vis_resized, CV_8UC3);
            else
                mask_vis.copyTo(mask_vis_resized);
            if (mask_vis_resized.size() != target_size)
                cv::resize(mask_vis_resized, mask_vis_resized, target_size);
                
            // ensure output is CV_8UC3 and correct size
            if (output_resized.type() != CV_8UC3)
                output_resized.convertTo(output_vis, CV_8UC3);
            else
                output_resized.copyTo(output_vis);
            if (output_vis.size() != target_size)
                cv::resize(output_vis, output_vis, target_size);

            // create empty panel with exact same size and type
            cv::Mat empty_panel = cv::Mat::zeros(target_size, CV_8UC3);

            // combine into five-panel debug image: 
            // row 1: original, blurred, flow mag (3 panels)
            // row 2: mask, result, empty (3 panels to match width)
            cv::Mat debug_row1, debug_row2, debug_image;
            
            // build row 1: frame + blurred + mag
            cv::hconcat(frame_vis, blurred_vis, debug_row1);
            cv::hconcat(debug_row1, mag_vis_resized, debug_row1);
            
            // build row 2: mask + output + empty
            cv::hconcat(mask_vis_resized, output_vis, debug_row2);
            cv::hconcat(debug_row2, empty_panel, debug_row2);
            
            // verify both rows have same width before vconcat
            if (debug_row1.cols != debug_row2.cols)
            {
                // resize row2 to match row1 width
                cv::resize(debug_row2, debug_row2, cv::Size(debug_row1.cols, debug_row2.rows));
            }
            
            // ensure both rows have same height
            if (debug_row1.rows != debug_row2.rows)
            {
                if (debug_row1.rows < debug_row2.rows)
                    cv::resize(debug_row1, debug_row1, cv::Size(debug_row1.cols, debug_row2.rows));
                else
                    cv::resize(debug_row2, debug_row2, cv::Size(debug_row2.cols, debug_row1.rows));
            }
            
            // stack rows vertically (now both have same width, height, and type)
            cv::vconcat(debug_row1, debug_row2, debug_image);

            // display results
            cv::imshow(kWinOut, output);
            cv::imshow(kWinDbg, debug_image);

            // save frame to output video if writer is open
            if (writer.isOpened())
            {
                writer.write(output);
            }
        }
        else
        {
            // first frame: just show original
            output = frame.clone();
            cv::imshow(kWinOut, output);
            
            // initialize debug window with empty panels (5 panels for Figure 2 layout)
            cv::Mat empty = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
            cv::Mat empty_row1, empty_row2, empty_panels;
            cv::hconcat(empty, empty, empty_row1);
            cv::hconcat(empty_row1, empty, empty_row1);
            cv::hconcat(empty, empty, empty_row2);
            cv::vconcat(empty_row1, empty_row2, empty_panels);
            cv::imshow(kWinDbg, empty_panels);
            
            first_frame = false;
        }

        // update previous frame
        gray.copyTo(prev_gray);

        // handle keyboard input
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q') // ESC or Q to quit
        {
            break;
        }
        else if (key == 's' || key == 'S') // S to save screenshot
        {
            std::string screenshot_name = "screenshot_" + std::to_string(frame_count) + ".png";
            cv::imwrite(screenshot_name, output);
            std::cout << "Screenshot saved: " << screenshot_name << std::endl;
        }

        frame_count++;
    }

    // cleanup
    cap.release();
    if (writer.isOpened())
    {
        writer.release();
    }
    cv::destroyAllWindows();

    return 0;
}
