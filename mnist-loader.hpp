#pragma once
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <tuple>
const std::string training_image_fn = "../mnist/train-images.idx3-ubyte";
const std::string training_label_fn = "../mnist/train-labels.idx1-ubyte";
const std::string test_image_fn = "../mnist/t10k-images.idx3-ubyte";
const std::string test_label_fn = "../mnist/t10k-labels.idx1-ubyte";
std::ifstream image;
std::ifstream label;

void print_image(Eigen::VectorXd im){
	for(int i = 0; i < 28*28 ; i++){
		if(i%28 == 0) std::cout << std::endl;
		std::cout << (im(i)==0?0:1);
	}
}

void init_mnist() {
	image.open(training_image_fn.c_str(), std::ios::in | std::ios::binary);
	label.open(training_label_fn.c_str(), std::ios::in | std::ios::binary);
		char number;
		for (int i = 1; i <= 16; ++i) {
			image.read(&number, sizeof(char));
		}
		for (int i = 1; i <= 8; ++i) {
			label.read(&number, sizeof(char));
		}
	}
std::tuple<Eigen::VectorXd, Eigen::VectorXd> get_image(){
	const int height = 28;
	const int width = 28;
	Eigen::VectorXd data(height*width);
	// Reading image
	char number;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			image.read(&number, sizeof(char));
			data(i*width+j) = (uint8_t)(number)/255.0;
		}
	}

	label.read(&number, sizeof(char));
	Eigen::VectorXd expected(10);
	for(int i = 0; i < 10; i++){
		if((int) number == i)
			expected(i) = 1.0;
		else
			expected(i) = 0.0;
	}
	print_image(data);
	std::cout << std::endl;
	std::cout << "label: " << (int) number << std::endl;
	std::tuple<Eigen::VectorXd, Eigen::VectorXd> res = std::make_tuple(data, expected);
	return res;
}
void mnist_test_init(){
	image.close();
	label.close();
	image.open(test_image_fn.c_str(), std::ios::in | std::ios::binary);
	label.open(test_label_fn.c_str(), std::ios::in | std::ios::binary);
		char number;
		for (int i = 1; i <= 16; ++i) {
			image.read(&number, sizeof(char));
		}
		for (int i = 1; i <= 8; ++i) {
			label.read(&number, sizeof(char));
		}
}
void close_mnist(){
	image.close();
	label.close();
}

