#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "mi_review_controller.h"
#include "util/mi_exception.h"
#include "log/mi_logger.h"

#ifdef WIN32
#else
#include <libgen.h>
#endif

using namespace medical_imaging;

int main(int argc , char* argv[]) {
    try {
#ifndef WIN32
        chdir(dirname(argv[0]));
#endif

        std::cout << "hello review\n";

        Logger::instance()->initialize();

        if (argc != 2) {
            std::cout << "invalid input\n";
            return -1;
        }

        std::string path(argv[1]);

        std::cout << "path is " << path << std::endl;
        std::shared_ptr<ReviewController> controller(new ReviewController());
        controller->initialize();
        controller->run(path);
    } catch (Exception& e) {
        std::cout << "Review server error exit : " << e.what() << std::endl;
    }

    std::cout << "Review server exit.\n";
    return 0;
}