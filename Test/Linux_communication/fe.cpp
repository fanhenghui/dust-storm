#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>

#include "common.h"

static std::mutex io_mutex;

void message_queue(int fd_remote)
{
    std::string in;
    while(std::cin >> in)
    {
        std::unique_lock<std::mutex> locker(io_mutex);

        Msg msg;
        msg.tag = 0;
        msg.len = in.size() + 1;
        msg.buffer = new char[msg.len];
        msg.buffer[msg.len-1] = '\0';
        for(int i = 0; i<msg.len-1 ; ++i)
        {
            msg.buffer[i] = in[i];
        }

        std::cout << "FE : " << msg.buffer << std::endl;

        if(-1 == send(fd_remote , &msg , sizeof(msg) , 0))
        {
            continue;
        }

        if(-1 == send(fd_remote , msg.buffer ,msg.len , 0))
        {
            continue;
        }

    }
}


void handle_command(int fd_remote ,  Msg& msg)
{
    if(0 == msg.tag)
    {
        std::cout << "BE : " << msg.buffer << std::endl;
    }
}

//主动启动一方，是进程间通讯的server方
int main(int argc , char* argv[])
{
    if(argc != 2){
        std::cout << "invalid input\n";        
        return -1;
    }

    std::cout << "path is " << argv[1] << std::endl;

    int fd_s = socket(AF_UNIX , SOCK_STREAM , 0);
    if(fd_s == -1){
        std::cout << "create socket failed!\n";
        return -1;
    }

    struct sockaddr_un local;
    local.sun_family = AF_UNIX;
    strcpy(local.sun_path , argv[1]);
    unlink(local.sun_path);

    if(bind(fd_s , (struct sockaddr*)(&local) , sizeof(local)) == -1){
        std::cout << "bind failed!\n";
        return -1;
    }

    if(listen(fd_s , 10) == -1){
        std::cout << "listen faild!\n";
        return -1;
    }

    struct sockaddr_un remote;
    socklen_t len = sizeof(remote);
    int fd_c = accept(fd_s , (struct sockaddr*)(&remote) , &len);
    if(fd_c == -1){
        std::cout << "accept failed!\n";
        return -1;
    }

    std::thread th(message_queue ,fd_c);
    th.detach();

    for(;;){
        Msg msg;
        if(-1 == recv(fd_c , &msg , sizeof(msg) , 0)){
            std::cout << "warning read failed!\n";
            continue;
        }
        int len = msg.len;
        if(len <= 0){
            continue;
        }

        msg.buffer = new char[len];
        if(-1 == recv(fd_c ,msg.buffer , len , 0) ){
            std::cout << "warning read failed!\n";
            continue;
        }

        handle_command(fd_c , msg);
    }


    return 0;

}