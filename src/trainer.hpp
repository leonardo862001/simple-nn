#include<Eigen/Core>
#include<vector>
#include "mnist-loader.hpp"
#include "network.hpp"
class trainer {
       public:
	trainer(network& net) : _net(net) {
		init_mnist();
		for( auto i : _net.w ){
			v.push_back(MatrixXd::Zero(i.rows(), i.cols()));
			dw.push_back(MatrixXd::Zero(i.rows(), i.cols()));
			db.push_back(VectorXd::Zero(i.rows()));
		}
	}
	~trainer() { close_mnist(); }
	void backpropagation(VectorXd expected) {
		std::vector<VectorXd> delta;
		for( auto i : _net.a ){
			delta.push_back(VectorXd::Zero(i.rows()));
		}
#ifdef SOFTMAX
		delta.back() = (dsoftmax(_net.a.back()).array() * (softmax(_net.a.back())-expected).array());
#else
		delta.back() = (dactivation(_net.a.back()).array() * (activation(_net.a.back())-expected).array());
#endif
		//delta.back() = (dsoftmax(_net.a.back()).array()) * (1-expected.array())/(1-softmax(_net.a.back()).array())-(expected.array())/(softmax(_net.a.back()).array());
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
				backpropagation(std::get<1>(im));
			}
			for(int k = 1; k < _net.w.size(); k++){
				_net.w[k] *= 1-eta*lambda/batch_size;
#ifdef clipping
				if(dw[k].norm() > threshold) dw[k] *= threshold/dw[k].norm();
				if(db[k].norm() > threshold) db[k] *= threshold/dw[k].norm();
#endif
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
	double eta = 0.42;
	double mu = 0.78;
	double lambda = 0.0015;
	double threshold = 0.1;
	int batch_size = 5;
	std::vector<MatrixXd> dw;
	std::vector<VectorXd> db;
	MatrixXd delta1, delta2;
	std::vector<MatrixXd> v;
};
