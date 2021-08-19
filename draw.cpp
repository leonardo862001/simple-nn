#include <SDL2/SDL.h>

#include <Eigen/Core>
#include <iostream>

#include "network.hpp"
#define WIDTH 224
#define HEIGHT 224
#define BRUSH 6
int main(int argc, char** argv) {
  network nn;
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
            int scale = WIDTH/28;
            float value=0;
            for(int i = 0; i < 28; i++){
              for(int j = 0; j < 28; j++){
                for(int k = 0; k < scale; k++){
                  for(int l = 0; l < scale; l++){
                    value+=pixels[(scale*i+k)*WIDTH+(scale*j)+l]&0xFF;
                  }
                }
                value/=scale*scale;
                value = 255-value;
                value/=255;
                x(i*28+j)=value;
                value = 0;
                std::cout << (x(i*28+j)==0?0:1);
              }
              std::cout << std::endl;
            }
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
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
