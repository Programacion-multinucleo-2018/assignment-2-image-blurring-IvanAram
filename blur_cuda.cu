#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include <cuda_runtime.h>

#define CONV_SIZE 5

using namespace std;

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height){
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if (xIndex < width && yIndex < height && xIndex >= 0 && yIndex >= 0){
		//Location of colored pixel in input
		const int tid = yIndex * (width * 3) + (3 * xIndex);

		const int step = (CONV_SIZE - 1) / 2;
		int idx;
		float sum_blue = 0.0f;
		float sum_green = 0.0f;
		float sum_red = 0.0f;
		float pixels = 0;
		for (size_t i = 0; i < CONV_SIZE; i++) {
			for (size_t j = 0; j < CONV_SIZE; j++) {
				idx = tid + ((i - step) * width * 3) + ((j - step) * 3);
				if(idx >= 0 && idx < (width * 3) * height && (xIndex * 3) >= 0 && xIndex < width) {
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

// Function to set up CUDA
void blur_image(const cv::Mat& input, cv::Mat& output){
	cout << "\nInput image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	size_t colorBytes = input.step * input.rows;

	// Variables to store the input and output images
	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, colorBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(32, 32);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("\nblur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	auto start_at = std::chrono::high_resolution_clock::now();
	// Launch the color conversion kernel
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows);
	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
	auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
	cout << "\nBlur image on gpu elapsed: " << duration_ms.count() << " ms" << endl;

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, colorBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
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

	//Create output image
	cv::Mat output(input.rows, input.cols, input.type());

	//Call the wrapper function
	blur_image(input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
