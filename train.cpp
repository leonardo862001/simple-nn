#include "network.hpp"
#include "trainer.hpp"
int main(){
	network nn({784, 50, 15, 10});
	trainer tr(nn);
	tr.learn();
	tr.test();
	nn.dump();
}
