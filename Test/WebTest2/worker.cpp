#include <iostream>
#include <unistd.h>

int main(int argc , char * argv[])
{
    if(argc > 1){

        std::cout << "worker : " << argv[1] << std::endl;
        while(true){

            sleep(1);
            static int tick = 0;
            std::cout << "tick " << argv[1] << tick++ << std::endl;
            if(tick > 20){
                break;
            }
        }
    }
    else{

        std::cout << "just worker\n";
    }
    
    return 0;
}
