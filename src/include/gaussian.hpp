#include<iostream>
#include<Eigen/Core>
#include<cmath>
using namespace Eigen;

double Gaussian2d(double x, double y, double sigma){
	return std::exp( -(x*x + y*y)/(2*sigma*sigma) )/(2*M_PI*sigma*sigma);
}
MatrixXd FilterCreation(double sigma, int size)
{
	MatrixXd kernel = MatrixXd::Zero(size, size);
	double sum = 0.0;
	int t = size / 2;
	for (int x = -t; x <= t; x++) {
		for (int y = -t; y <= t; y++) {
			kernel(x + t, y + t) = Gaussian2d(x, y, sigma);
			sum += kernel(x + t,y + t);
		}
	}

	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			kernel(i, j) /= sum;
	return kernel;
}
