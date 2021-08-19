#include<iostream>
#include<fstream>
#include<Eigen/Core>
#include "network.hpp"
using namespace Eigen;

int main(int argc, char** argv){
	network nn;
	nn.load();
	std::ifstream image;
	image.open(argv[1], std::ios::in);
	VectorXd im;
	int rows, cols, bits;
	char number;
	image >> number; //P
	image >> number; //2 ASCII formatted pgm
	char buf[100];
	image.getline(buf, 100);
	image.getline(buf, 100);//gimp comment
	image >> rows;
	image >> cols;
	image >> bits; //grayscale bit
	im = VectorXd::Zero(rows*cols);
	for(int i = 0; i < rows*cols; i++){
		image >> im(i);
		im(i)/=255;
		if(i%rows==0) std::cout << std::endl;
		std::cout << ((im(i)==0)?0:1);
	}
	image.close();
	VectorXd y = nn.feedforward(im);
	int max_i = 0;
	for(int i = 0; i < 10; i++)
				if(y(i) > y(max_i)) max_i = i;
	std::cout << "\nresult: " << max_i << std::endl;
	return 0;
}
