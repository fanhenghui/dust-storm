#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "mi_review_controller.h"
#include "MedImgUtil/mi_exception.h"

using namespace medical_imaging;

int main(int argc , char* argv[])
{
    try
    {
        std::cout << "hello review\n";
        if(argc != 2) {
            std::cout << "invalid input\n";
            return -1;
        }
        std::string path(argv[1]);

        std::cout << "path is " << path << std::endl;
        std::shared_ptr<ReviewController> controller(new ReviewController());
        controller->initialize();
        controller->run(path);
    }
    catch(Exception& e)
    {
        std::cout << "Review server error exit : " << e.what() << std::endl;
    } 

    std::cout << "Review server exit.\n"; 
    return 0;
}