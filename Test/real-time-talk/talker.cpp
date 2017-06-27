#include <iostream>
#include <unistd.h>

int main(int argc , char* argv[])
{
    if(argc > 1){

        std::cout << "talker : " << argv[1] << std::endl;
        while(true){

            sleep(1);
            static int tick = 0;
            std::cout << "tick : " << argv[1] << "\t" << tick++ << std::endl;
            if(tick > 100){
                break;
            }
        }
    }
    else{

        std::cout << "just talker\n";
    }
    
    return 0;
}