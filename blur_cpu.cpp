#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

// Constant that stores convolutional matrix size
#define CONV_SIZE 5

using namespace std;

// Main function that blurs image
void blur_image(unsigned char* input, unsigned char* output, int width, int height){
	const int step = (CONV_SIZE - 1) / 2;
	int idx, tid;
	float sum_blue;
	float sum_green;
	float sum_red;
	float pixels;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			tid = row * (width * 3) + (col * 3);
			sum_blue = 0.f;
			sum_green = 0.f;
			sum_red = 0.f;
			pixels = 0;
			for (size_t i = 0; i < CONV_SIZE; i++) {
				for (size_t j = 0; j < CONV_SIZE; j++) {
					idx = tid + ((i - step) * width * 3) + ((j - step) * 3);
					if(idx >= 0 && idx < (width * 3) * height) {
						sum_blue += input[idx];
						sum_green += input[idx + 1];
						sum_red += input[idx + 2];
						pixels += 1;
					}
				}
			}
			output[tid] = static_cast<unsigned char>(sum_blue * (1 / pixels));
			output[tid + 1] = static_cast<unsigned char>(sum_green * (1 / pixels));
			output[tid + 2] = static_cast<unsigned char>(sum_red * (1 / pixels));
		}
	}
}

int main(int argc, char *argv[]){
	string imagePath;

	if(argc < 2)
		imagePath = "image.jpg";
	else
		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	// Create output image
	cv::Mat output(input.rows, input.cols, input.type());

	// Blur image on host without threads
	auto start_at = std::chrono::high_resolution_clock::now();
	blur_image(input.ptr(), output.ptr(), input.cols, input.rows);
	auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  cout << "\nBlur image on host without threads elapsed: " << duration_ms.count() << " ms" << endl;

	// Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
