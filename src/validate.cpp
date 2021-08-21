#include "include/network.hpp"
#include "include/trainer.hpp"
int main(){
	network nn;
	nn.load();
	trainer tr(nn);
	tr.test();
}
