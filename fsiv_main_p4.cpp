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
     if (fps <= 0) fps = 30.0; // default fps
 
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
 
             // create debug visualization (three panels: flow mag, edges, mask)
             cv::Mat mag_vis, edges_vis, mask_vis;
             mag.convertTo(mag_vis, CV_8U, 255.0 / 10.0); // normalize for visualization
             edges.copyTo(edges_vis);
             refined_mask.copyTo(mask_vis);
 
             // combine into three-panel debug image
             cv::Mat debug_panel1, debug_panel2, debug_panel3;
             cv::cvtColor(mag_vis, debug_panel1, cv::COLOR_GRAY2BGR);
             cv::cvtColor(edges_vis, debug_panel2, cv::COLOR_GRAY2BGR);
             cv::cvtColor(mask_vis, debug_panel3, cv::COLOR_GRAY2BGR);
 
             cv::Mat debug_top, debug_bottom, debug_image;
             cv::hconcat(debug_panel1, debug_panel2, debug_top);
             cv::hconcat(debug_top, debug_panel3, debug_image);
 
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
             // show original
             output = frame.clone();
             cv::imshow(kWinOut, output);
             
             // initialize debug window with empty panels
             cv::Mat empty = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
             cv::Mat empty_triple;
             cv::hconcat(empty, empty, empty_triple);
             cv::hconcat(empty_triple, empty, empty_triple);
             cv::imshow(kWinDbg, empty_triple);
             
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
         else if (key == 's' || key == 'S') // S to save screenshot and add frame number to filename to avoid dublicating
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