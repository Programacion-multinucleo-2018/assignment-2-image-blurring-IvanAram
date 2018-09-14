#include <iostream>
#include <cstdio>
#include <pthread.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Constant that stores convolutional matrix size
#define CONV_SIZE 5
// Number of procesors
#define THREADS 8

// Type definition of data passed to threads
typedef struct thread_data_struct {
	unsigned char *input;
	unsigned char *output;
	int width;
	int height;
	int start;
	int end;
} thread_data_t;

using namespace std;

// Function to blur image used by threads
void *blur_image_threads(void *args){
	thread_data_t *data = (thread_data_t *) args;
	const int step = (CONV_SIZE - 1) / 2;
	int tid, idx;
	float sum_blue = 0.f;
	float sum_green = 0.f;
	float sum_red = 0.f;
	float pixels = 0;
	for (int row = data->start; row < data->end; row++) {
		for (int col = 0; col < data->width; col++) {
			tid = row * (data->width * 3) + (col * 3);
			sum_blue = 0.f;
			sum_green = 0.f;
			sum_red = 0.f;
			pixels = 0;
			for (size_t i = 0; i < CONV_SIZE; i++) {
				for (size_t j = 0; j < CONV_SIZE; j++) {
					idx = tid + ((i - step) * data->width * 3) + ((j - step) * 3);
					if(idx >= 0 && idx < (data->width * 3) * data->height) {
						sum_blue += data->input[idx];
						sum_green += data->input[idx + 1];
						sum_red += data->input[idx + 2];
						pixels += 1;
					}
				}
			}
			data->output[tid] = static_cast<unsigned char>(sum_blue * (1 / pixels));
			data->output[tid + 1] = static_cast<unsigned char>(sum_green * (1 / pixels));
			data->output[tid + 2] = static_cast<unsigned char>(sum_red * (1 / pixels));
		}
	}
	pthread_exit(NULL);
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

	// Declare threads and thread data
  pthread_t threads[THREADS];
  thread_data_t data[THREADS];

	// Initialize threads data
  int step = (int)(input.rows / THREADS);
  for (size_t i = 0; i < THREADS; i++) {
    data[i].start = step * i;
    data[i].end = step * (i + 1);
    data[i].width = input.cols;
		data[i].height = input.rows;
    data[i].input = input.ptr();
    data[i].output = output.ptr();
  }

	// Blur image on host with threads
	auto start_at = std::chrono::high_resolution_clock::now();
	// Create threads
	for (size_t i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, blur_image_threads, (void *) &data[i]);
  }
	// Wait for threads to finish
  for (size_t i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
	auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  cout << "\nBlur image on host with threads elapsed: " << duration_ms.count() << " ms" << endl;

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
