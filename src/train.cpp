#include "network.hpp"
#include "trainer.hpp"
int main(){
	//activation = SeLU;
	//dactivation = dSeLU;
	network nn({784, 50, 10});
	trainer tr(nn);
	tr.threshold = 0.02;
	tr.eta = 0.32; //0.65
	tr.mu = 0.74; //0.8
	tr.lambda = 0.0011; //0.002
	tr.learn();
	tr.test();
	nn.dump();
}
