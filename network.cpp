#include<Eigen/Core>
#include<initializer_list>
#include<vector>

#include "mnist-loader.hpp"
using namespace Eigen;

VectorXd sigmoid(VectorXd x) { return 1 / (1 + ((-x).array()).exp()); }
VectorXd dsigmoid(VectorXd x) { return sigmoid(x).array()*(1-sigmoid(x).array()); }
VectorXd ReLU(VectorXd x) { auto f = [](double k)->double{ return k<0?0:k; }; return x.unaryExpr(f); }
VectorXd dReLU(VectorXd x) { auto f = [](double k)->double{ return k<=0?0:1; }; return x.unaryExpr(f); }
VectorXd htangent(VectorXd x) { return x.array().tanh(); }
VectorXd dhtangent(VectorXd x) { return 1-((x.array().tanh()).square()); }
auto activation = sigmoid;
auto dactivation = dsigmoid;
class network {
       public:
	network(std::initializer_list<int> l) {
		for( auto i : l ){
			b.push_back(VectorXd::Zero(i));
			a.push_back(VectorXd::Zero(i));
		}
		w.push_back(MatrixXd::Zero(1,1));
		for(int i = 1; i < a.size(); i++){
			w.push_back(MatrixXd::Random(a[i].rows(), a[i-1].rows()));
		}
	}
	VectorXd feedforward(VectorXd input) {
		a[0] = input;
		int i;
		a[1] = w[1]*a[0]+b[1];
		for(i = 1; i < a.size()-1; i++){
			a[i+1] = w[i+1] * activation(a[i])+b[i+1];
		}
		return activation(a[i]);
	}
	std::vector<MatrixXd> w;
	std::vector<VectorXd> a;
	std::vector<VectorXd> b;
};

class trainer {
       public:
	trainer(network& net) : _net(net) {
		init_minst();
		for( auto i : _net.w ){
			v.push_back(MatrixXd::Zero(i.rows(), i.cols()));
			dw.push_back(MatrixXd::Zero(i.rows(), i.cols()));
			db.push_back(VectorXd::Zero(i.rows()));
		}
	}
	~trainer() { close_minst(); }
	void backpropagation(VectorXd expected, VectorXd input) {
		std::vector<VectorXd> delta;
		for( auto i : _net.a ){
			delta.push_back(VectorXd::Zero(i.rows()));
		}
		delta.back() = (dactivation(_net.a.back()).array() * (activation(_net.a.back())-expected).array());
		v.back() = mu*v.back()- eta*delta.back()*activation(_net.a[delta.size()-2]).transpose();
		//_net.w.back() *= 1-eta*lambda;
		dw.back() += v.back();
		db.back() -= eta*delta.back();
		for(int i = delta.size()-2; i > 0; i--){
			delta[i] = (_net.w[i+1].transpose()*delta[i+1]).array()*dactivation(_net.a[i]).array();
			v[i] =mu*v[i] - eta*delta[i]*activation(_net.a[i-1]).transpose();
			//_net.w[i] *= 1-eta*lambda;
			dw[i] += v[i];
			db[i] -= eta*delta[i];
		}
	}
	void learn(){
		for(int i = 0; i < 60000/batch_size; i++){
			for(int k = 1; k < _net.w.size(); k++){
				dw[k] = MatrixXd::Zero(_net.w[k].rows(), _net.w[k].cols());
				db[k] = VectorXd::Zero(_net.b[k].rows());
			}
			std::cout << "Batch: " << i << std::endl;
			for(int j=0; j < batch_size; j++){
				std::cout << "Sample: " << j << std::endl;
				auto im = get_image();
				VectorXd y = _net.feedforward(std::get<0>(im));
				std::cout << y.transpose() << std::endl;
				backpropagation(std::get<1>(im),std::get<0>(im));
			}
			for(int k = 1; k < _net.w.size(); k++){
				_net.w[k] *= 1-eta*lambda/batch_size;
				_net.w[k] += dw[k]/batch_size;
				_net.b[k] += db[k]/batch_size;
			}
		}
	}
	void test(){
		mnist_test_init();
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
		std::cout << "Accuracy: " << ok/10000 *100<< "%" << std::endl;
	}
	network& _net;
	const double eta = 0.42;
	const double mu = 0.78;
	const double lambda = 0.0015;
	const int batch_size = 5;
	std::vector<MatrixXd> dw;
	std::vector<VectorXd> db;
	MatrixXd delta1, delta2;
	std::vector<MatrixXd> v;
};

int main(){
	network nn({784, 50, 15, 10});
	trainer tr(nn);
	tr.learn();
	tr.test();
}
