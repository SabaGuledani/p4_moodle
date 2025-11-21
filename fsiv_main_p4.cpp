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

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, KEYS);
    parser.about("FSIV Smart Background Filter (motion + edges)");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }


    // GUI setup
    const std::string kWinOut = "FSIV Output";
    const std::string kWinDbg = "FSIV Debug (flow|edges|mask)";

    while (true)
    {
        if (!cap.read(frame) || frame.empty())
            break;


        }


    }

    return 0;
}
