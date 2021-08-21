#include<Eigen/Core>
using namespace Eigen;

MatrixXd conv2d(MatrixXd image, MatrixXd kernel){
	MatrixXd ret = MatrixXd::Zero(image.rows(), image.cols());
	int s = kernel.rows()/2;
	for(int i = s; i < image.rows() - s; i++)
		for(int j = s; j < image.cols() - s; j++)
			for(int k = -s; k <= s; k++)
				for(int l = -s; l <= s; l++)
					ret(i,j)+=image(i+k,j+l)*kernel(k+s,l+s);

	double norm = kernel.sum();
	for(int i = 0; i < image.rows(); i++){
		for(int j = 0; j < s; j++){
			double sum = 0.0;
			for(int k = -s; k <= s; k++){
				for(int l = -s; l <= s; l++){
					if( i+k < 0 || i+k >= image.rows() || j+l < 0 || j+l >= image.cols() ) sum+=kernel(k+s,l+s);
					else ret(i, j)+=image(i+k,j+l)*kernel(k+s,l+s);
				}
			}
			ret(i,j)*=norm/(norm-sum);
		}
	}
	for(int i = 0; i < image.rows(); i++){
		for(int j = image.cols() - s; j < image.cols(); j++){
			double sum = 0.0;
			for(int k = -s; k <= s; k++){
				for(int l = -s; l <= s; l++){
					if( i+k < 0 || i+k >= image.rows() || j+l < 0 || j+l >= image.cols()) sum+=kernel(k+s,l+s);
					else ret(i, j)+=image(i+k,j+l)*kernel(k+s,l+s);
				}
			}
			ret(i,j)*=norm/(norm-sum);
		}
	}
	for(int i = 0; i < s; i++){
		for(int j = s; j < image.cols()-s; j++){
			double sum = 0.0;
			for(int k = -s; k <= s; k++){
				for(int l = -s; l <= s; l++){
					if( i+k < 0 || i+k > image.rows() || j+l < 0 || j+l >= image.cols()) sum+=kernel(k+s,l+s);
					else ret(i, j)+=image(i+k,j+l)*kernel(k+s,l+s);
				}
			}
			ret(i,j)*=norm/(norm-sum);
		}
	}
	for(int i = image.rows() - s; i < image.rows(); i++){
		for(int j = s; j < image.cols()-s; j++){
			double sum = 0.0;
			for(int k = -s; k <= s; k++){
				for(int l = -s; l <= s; l++){
					if( i+k < 0 || i+k >= image.rows() || j+l < 0 || j+l >= image.cols()) sum+=kernel(k+s,l+s);
					else ret(i, j)+=image(i+k,j+l)*kernel(k+s,l+s);
				}
			}
			ret(i,j)*=norm/(norm-sum);
		}
	}
	return ret;
}
