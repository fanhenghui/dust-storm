#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

int main(int argc , char* argv[])
{
    std::cout << "main start>>>\n";
    
    
    //1 System 
    //  printf("running ps with system\n");
    //  system("./run &");
    //  printf("ps Done\n");

    //2exec
    //execl("./print_msg" , "" , "" , 0);

    //3 fork
    pid_t pid = fork();
    std::cout << pid << std::endl;
    switch(pid)
    {
        case -1:
            std::cout << "fork failed!\n";
            exit(1);
        case 0:
            std::cout << "in child process fork success!\n";
            execl("./run" , "./run" , "fork &" , 0);
            break;
        default:
            std::cout << "in parent\n";
            break;
    } 

    std::cout << "main quit>>>\n";
    return 0;
}