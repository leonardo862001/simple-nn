#include <SDL2/SDL.h>

#include <Eigen/Core>
#include <iostream>
#include <fstream>

#include "network.hpp"
#include "trainer.hpp"
#define WIDTH 224
#define HEIGHT 224
#define BRUSH 5

MatrixXd reduce(MatrixXd x){
  Matrix3d kernel;
  kernel << 1,2,1,
         2,4,2,
         1,2,1;
  kernel/=16;
  MatrixXd ret = MatrixXd::Zero(x.rows()/2, x.cols()/2);
  MatrixXd temp = MatrixXd::Zero(x.rows(), x.cols());
  for(int i = 1; i < x.rows() - 1; i++){
    for(int j = 1; j < x.cols() - 1; j++){
      for(int k = -1; k <= 1; k++){
        for(int l = -1; l <= 1; l++){
          temp(i,j)+=x(i+k,j+l)*kernel(1+k, 1+l);
        }
      }
    }
  }
  for(int i = 0; i < x.rows()/2; i++){
    for(int j = 0; j < x.cols()/2; j++){
      ret(i,j)=temp(2*i+1,2*j+1)+temp(2*i+1,2*j)+temp(2*i,2*j+1)+temp(2*i,2*j);
      ret(i,j)/=4;
    }
  }
  return ret;
}


int main(int argc, char** argv) {
  network nn;
  trainer tr(nn);
  tr.eta = 0.32;
  tr.mu = 0.7;
  tr.lambda = 0.0011;
  tr.batch_size = 1;
  nn.load();
  VectorXd x = VectorXd::Zero(28 * 28);
  bool quit = false;
  bool leftMouseButtonDown = false;
  SDL_Event event;
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window* window = SDL_CreateWindow("Test", SDL_WINDOWPOS_UNDEFINED,
					SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);
  SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
  SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
					   SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);
  uint32_t* pixels = new uint32_t[WIDTH*HEIGHT];
  memset(pixels, 255, WIDTH * HEIGHT * sizeof(uint32_t));
  while (!quit) {
    SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(uint32_t));
    SDL_WaitEvent(&event);
    switch (event.type) {
      case SDL_QUIT:
				quit = true;
				break;
      case SDL_KEYDOWN:
        switch(event.key.keysym.sym){
          case SDLK_KP_ENTER:
            {
            MatrixXd temp = MatrixXd::Zero(WIDTH, HEIGHT);
            for(int i = 0; i < WIDTH; i++){
              for(int j = 0; j < HEIGHT; j++){
                temp(i,j) = pixels[i*WIDTH+j]&0xFF;
              }
            }
            temp = 255-temp.array();
            temp /= 255;
            int scale = WIDTH/28;
#ifndef  GAUSSIAN_SCALING
#ifdef OUT
            std::ofstream file;
            file.open("original.pgm", std::ios::out);
            file << "P2\n";
            file << temp.rows() << " " << temp.cols() << "\n";
            file << "255\n";
            file << temp*255;
            file.close()
#endif
            float value=0;
            for(int i = 0; i < 28; i++){
              for(int j = 0; j < 28; j++){
                for(int k = 0; k < scale; k++){
                  for(int l = 0; l < scale; l++){
                    value+=temp((scale*i+k),(scale*j)+l);
                  }
                }
                value/=scale*scale;
                x(i*28+j)=value;
                value = 0;
                std::cout << (x(i*28+j)==0?0:1);
              }
              std::cout << std::endl;
            }
#ifdef OUT
            file.open("reduction.pgm", std::ios::out);
            file << "P2\n";
            file << 28 << " " << 28 << "\n";
            file << "255\n";
            file << x*255;
            file.close();
#endif
#else
            MatrixXd t;
#ifdef OUT
            std::ofstream file;
            file.open("original.pgm", std::ios::out);
            file << "P2\n";
            file << temp.rows() << " " << temp.cols() << "\n";
            file << "255\n";
            file << temp*255;
            file.close();
#endif
            for(int i = 0; i < 3; i++){
              t = reduce(temp);
              temp.resize(temp.rows()/2, temp.cols()/2);
              temp = t;
              t.resize(t.rows()/2, t.cols()/2);
#ifdef OUT
              file.open("reduction"+std::to_string(i)+".pgm", std::ios::out);
              file << "P2\n";
              file << temp.rows() << " " << temp.cols() << "\n";
              file << "255\n";
              file << temp*255;
              file.close();
#endif
            }
            for(int i = 0; i < 28; i++){
              for(int j = 0; j < 28; j++){
                x(i*28+j) = temp(i,j);
                std::cout << (x(i*28+j)==0?0:1);
              }
              std::cout << std::endl;
            }
#endif
            	VectorXd y = nn.feedforward(x);
	            int max_i = 0;
	            for(int i = 0; i < 10; i++)
				      if(y(i) > y(max_i)) max_i = i;
	            std::cout << "\nresult: " << max_i << std::endl;
              break;
            }
          case SDLK_DELETE:
            memset(pixels, 255, WIDTH * HEIGHT * sizeof(uint32_t));
				    break;
          case SDLK_KP_0:
            tr.backpropagation( (VectorXd(10) << 1,0,0,0,0,0,0,0,0,0).finished());
				    break;
          case SDLK_KP_1:
            tr.backpropagation( (VectorXd(10) << 0,1,0,0,0,0,0,0,0,0).finished());
				    break;
          case SDLK_KP_2:
            tr.backpropagation( (VectorXd(10) << 0,0,1,0,0,0,0,0,0,0).finished());
				    break;
          case SDLK_KP_3:
            tr.backpropagation( (VectorXd(10) << 0,0,0,1,0,0,0,0,0,0).finished());
				    break;
          case SDLK_KP_4:
            tr.backpropagation( (VectorXd(10) << 0,0,0,0,1,0,0,0,0,0).finished());
				    break;
          case SDLK_KP_5:
            tr.backpropagation( (VectorXd(10) << 0,0,0,0,0,1,0,0,0,0).finished());
				    break;
          case SDLK_KP_6:
            tr.backpropagation( (VectorXd(10) << 0,0,0,0,0,0,1,0,0,0).finished());
				    break;
          case SDLK_KP_7:
            tr.backpropagation( (VectorXd(10) << 0,0,0,0,0,0,0,1,0,0).finished());
				    break;
          case SDLK_KP_8:
            tr.backpropagation( (VectorXd(10) << 0,0,0,0,0,0,0,0,1,0).finished());
				    break;
          case SDLK_KP_9:
            tr.backpropagation( (VectorXd(10) << 0,0,0,0,0,0,0,0,0,1).finished());
				    break;
        }
    	case SDL_MOUSEBUTTONUP:
				if (event.button.button == SDL_BUTTON_LEFT) leftMouseButtonDown = false;
				break;
      case SDL_MOUSEBUTTONDOWN:
				if (event.button.button == SDL_BUTTON_LEFT) leftMouseButtonDown = true;
      case SDL_MOUSEMOTION:
				if (leftMouseButtonDown) {
	  			int mouseX = event.motion.x;
	  			int mouseY = event.motion.y;
          for(int i = -BRUSH; i <= BRUSH; i++){
            for(int j = -BRUSH; j <= BRUSH; j++){
	  			    pixels[(mouseY+i) * WIDTH + mouseX+j] = 0;
            }
          }
				}
				break;
    }
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
  }
  delete[] pixels;
  nn.dump();
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
