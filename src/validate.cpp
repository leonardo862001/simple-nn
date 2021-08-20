#include "network.hpp"
#include "trainer.hpp"
int main(){
	network nn;
	nn.load();
	trainer tr(nn);
	tr.test();
}
