#include "io/mi_db.h"
#include "io/mi_io_logger.h"
#include "io/mi_configure.h"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>

#ifdef WIN32
#else
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#endif

int db_ut(int argc, char* argv[]) {
#ifndef WIN32
    if(0 != chdir(dirname(argv[0]))) {

    }
#endif

    



    return 0;
}