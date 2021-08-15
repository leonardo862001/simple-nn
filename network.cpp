#include <Eigen/Core>

#include "mnist-loader.hpp"
using namespace Eigen;

VectorXd sigmoid(VectorXd x) { return 1 / (1 + ((-x).array()).exp()); }
class network {
       public:
	network(int i, int h, int o) {
		w0 = MatrixXd::Random(h, i);
		w1 = MatrixXd::Random(o, h);
		b0 = VectorXd::Zero(h);
		b1 = VectorXd::Zero(o);
	}
	VectorXd feedforward(VectorXd input) {
		a1 = w0 * input + b0;
		a1 = sigmoid(a1);
		a2 = w1 * a1 + b1;
		a2 = sigmoid(a2);
		return a2;
	}
	MatrixXd w0, w1;
	VectorXd a1, a2;
	VectorXd b0, b1;
};

class trainer {
       public:
	trainer(network& net) : _net(net) {
		init_minst();
	}
	~trainer() { close_minst(); }
	float cost(VectorXd y, VectorXd expected) {
		return 0.5 * (expected - y).squaredNorm();
	}
	float batch_cost(int batch_size, VectorXd* inputs, VectorXd* outputs) {
		float e = 0;
		for (int i = 0; i < batch_size; i++) {
			e += cost(outputs[i], inputs[i]);
		}
		return e / batch_size;
	}
	void backpropagation(VectorXd expected, VectorXd input) {
		VectorXd yd(expected.rows());
		yd = _net.a2.array() * (1-_net.a2.array()) * (_net.a2-expected).array();
		VectorXd hd(_net.a1.rows());
		hd = (_net.w1.transpose()*yd).array()*(_net.a1.array() * (1-_net.a1.array())).array();
		delta2 = yd*_net.a1.transpose();
		_net.w1 -= eta*delta2;
		_net.b1 -= eta*yd;
		delta1 = hd*input.transpose();
		_net.w0 -= eta*delta1;
		_net.b0 -= eta*hd;
	}
	void learn(){
		for(int i = 0; i < 500; i++){
			std::cout << "Batch: " << i << std::endl;
			for(int j=0; j < 100; j++){
				std::cout << "Sample: " << j << std::endl;
				auto im = get_image();
				VectorXd y = _net.feedforward(std::get<0>(im));
				std::cout << y.transpose() << std::endl;
				backpropagation(std::get<1>(im),std::get<0>(im));
			}
		}
	}
	void test(){
		std::cout << "testing: " << std::endl;
		double ok = 0.0;
		for(int i = 0; i < 10000; i++){
			auto x = get_image();
			VectorXd y = _net.feedforward(std::get<0>(x));
			int max_i = 0;
			for(int i = 0; i < 10; i++){
				if(y(i) > y(max_i)) max_i = i;
			}
			int max_i_c = 0;
			for(int i = 0; i < 10; i++){
				if(std::get<1>(x)(i) > std::get<1>(x)(max_i_c)) max_i_c = i;
			}
			std::cout << "\nresult: " << max_i << std::endl;
			if(max_i == max_i_c) ok+=1;

		}
		std::cout << "Accuracy: " << ok/10000 << std::endl;
	}
	network& _net;
	const double eta = 0.32;
	const double momentum = 0.9;
	MatrixXd delta1, delta2;
};

int main(){
	network nn(784, 30, 10);
	trainer tr(nn);
	tr.learn();
	tr.test();
}
