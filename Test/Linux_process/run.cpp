#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <fstream>

int main(int argc , char* argv[])
{
    printf("running process\n");
    std::string file_id = "0";
    if(argc == 2)
    {
        file_id = std::string(argv[1]);
    }
    std::string file_name = std::string("log_") + file_id;
    std::ofstream out(file_name.c_str(), std::ios::out);
    if(!out.is_open())
    {
        std::cout << "Open file " << file_name << " failed.\n";
        return -1;
    }

    while(true)
    {
        // std::string in;
        // std::cin >> in;
        // if(in == "exit")
        // {
        //     break;
        // }
        static int x;
        sleep(1);
        out << "tick " << x++ << std::endl;
    }
    out.close();
    printf("running done\n");
    return 0;
}