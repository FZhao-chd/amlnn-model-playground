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



#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "nn_sdk.h"
#include "model_loader.h"
#include "postprocess.h"

const std::string DEFAULT_OUTPUT_PATH = "result.jpg";
const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;

const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model_path> <image_path> [output_path]\n", argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : DEFAULT_OUTPUT_PATH;

    printf("Model: %s\n", model_path.c_str());
    printf("Image: %s\n", image_path.c_str());

    // 1. Initialize Network
    void* ctx = init_network(model_path.c_str());
    if (!ctx) {
        fprintf(stderr, "Failed to initialize network\n");
        return -1;
    }

    // 2. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        uninit_network(ctx);
        return -1;
    }

    // 3. Preprocess
    auto start_time = std::chrono::high_resolution_clock::now();
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple = preprocess(img, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));

    // 4. Inference
    nn_output* outdata = (nn_output*)run_network(ctx, {input_tuple});
    if (!outdata) {
        fprintf(stderr, "Inference failed\n");
        uninit_network(ctx);
        return -1;
    }

    // 5. Postprocess
    float* out0 = (float*)outdata->out[0].buf;
    float* out1 = (float*)outdata->out[1].buf;
    float* out2 = (float*)outdata->out[2].buf;

    int num_classes = CLASS_NAMES.size();
    int channels = num_classes + 64; 
    
    std::vector<Detection> detections = postprocess(
        std::make_tuple(out0, std::make_tuple(MODEL_INPUT_HEIGHT / 16, MODEL_INPUT_WIDTH / 16, channels), 16),
        std::make_tuple(out1, std::make_tuple(MODEL_INPUT_HEIGHT / 8, MODEL_INPUT_WIDTH / 8, channels), 8),
        std::make_tuple(out2, std::make_tuple(MODEL_INPUT_HEIGHT / 32, MODEL_INPUT_WIDTH / 32, channels), 32),
        input_tuple, SCORE_THRESHOLD, NMS_THRESHOLD, num_classes, 1
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
    printf("Inference + Postprocess time: %.2f ms\n", inference_time.count());
    printf("Detections: %zu\n", detections.size());

    // 6. Draw and Save
    cv::Mat res = draw_detections(img, detections, CLASS_NAMES);
    cv::imwrite(output_path, res);
    printf("Saved result to %s\n", output_path.c_str());

    // 7. Cleanup
    uninit_network(ctx);

    return 0;
}
