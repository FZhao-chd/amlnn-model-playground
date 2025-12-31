/*
 * Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _AMLNN_YOLOV8_DEMO_POSTPROCESS_H_
#define _AMLNN_YOLOV8_DEMO_POSTPROCESS_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <unordered_set>

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} Detection;

// Function declarations
// Function declarations

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

std::vector<Detection> postprocess(std::tuple<float*, std::tuple<int, int, int>, int> out0,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out1,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out2,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold, int num_classes, int reverse);


cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections, 
                        const std::vector<std::string>& classes);

#endif //_AMLNN_YOLOV8_DEMO_POSTPROCESS_H_
