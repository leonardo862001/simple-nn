#pragma once
#include<Eigen/Core>
#include<initializer_list>
#include<vector>
#include<fstream>

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
	network() {}
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
	void dump(){
		std::ofstream file;
		file.open("data.dat", std::ios::out);
		file << w.size() <<  std::endl;
		for(int i = 1; i < w.size(); i++){
			file << w[i].rows() << " " << w[i].cols() << std::endl;
			file << w[i];
			file << std::endl;
			file << b[i].rows() << " " << b[i].cols() << std::endl;
			file << b[i].transpose();
			file << std::endl;
		}
		file.close();
	}
	void load(){
		std::ifstream file;
		file.open("data.dat", std::ios::in);
		int len;
		file >> len;
		w.push_back(MatrixXd::Zero(1,1));
		b.push_back(VectorXd::Zero(1));
		a.push_back(VectorXd::Zero(1));
		int rows, cols;
		for(int i = 1; i < len; i++){
			file >> rows >> cols;
			w.push_back(MatrixXd::Zero(rows, cols));
			for(int j = 0; j < rows; j++){
				for(int k = 0; k < cols; k++){
					file >> w[i](j,k);
				}
			}
			file >> rows >> cols;
			b.push_back(VectorXd::Zero(rows));
			a.push_back(VectorXd::Zero(rows));
			for(int j = 0; j < rows; j++){
				file >> b[i](j);
			}
		}
	}
	std::vector<MatrixXd> w;
	std::vector<VectorXd> a;
	std::vector<VectorXd> b;
};


