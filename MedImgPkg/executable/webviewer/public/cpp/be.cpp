#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "common.h"
#include "img_gen.h"

static std::mutex io_mutex;

//#define TEST_IMG

// std::shared_ptr<ImgGen> _imgGen;
int width = 512;
int height = 512;

std::shared_ptr<ImgSeqGen> _imgGen;
int deep = 713;
std::string data_path = "/home/zhangchanggong/data/AB_CTA_01.raw";

void message_queue(int fd_remote) {
    /////////////////////////////////////////////////////////
    // Test 3 sequace image
    if (!_imgGen) {
        _imgGen.reset(new ImgSeqGen());
        _imgGen->set_raw_data(data_path, width, height, deep);
    }

    int tick = 0;

    while (true) {
        ++tick;

        if (tick == 1000000) {
            return;
        }

        static int slice = 0;

        if (slice > deep - 1) {
            slice = 0;
        }

        // 1秒触发一次
        usleep(50000);

        Msg msg;
        msg.tag = 1;
        msg.len = width * height * 4;
        msg.buffer = (char*)(_imgGen->gen_img(slice++));
        char* buffer = new char[msg.len + 16];
        memcpy(buffer, &(msg), 16);
        memcpy(buffer + 16, msg.buffer, msg.len);

        send(fd_remote, buffer, msg.len + 16, 0);

        delete[] buffer;
    }

    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    // Test 2 black image
    // if(!_imgGen){
    //     _imgGen.reset(new ImgGen());
    // }
    // int tick =0;
    // while(true){
    //     ++tick;
    //     if(tick == 1000000){
    //         return;
    //     }

    //     //1秒触发一次
    //     usleep(50000);

    //     Msg msg;
    //     msg.tag = 1;
    //     msg.len = width*height*4;
    //     msg.buffer = (char*)(_imgGen->gen_img(width,height));
    //     char* buffer = new char[msg.len + 16];
    //     memcpy(buffer , &(msg) , 16);
    //     memcpy(buffer+16 , msg.buffer , msg.len);

    //     send(fd_remote , buffer ,msg.len+16 , 0);

    //     delete [] buffer;
    // }
    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    // Test 1 string message
    // int tick =0;
    // while(true)
    // {
    //     ++tick;
    //     //100次循环发送一个心跳
    //     sleep(2);
    //     std::cout << "tick " << std::endl;
    //     std::string msg_str = std::string("I am still on!");
    //     Msg msg;
    //     msg.tag = 0;
    //     msg.len = msg_str.size() + 1;
    //     msg.buffer = new char[msg.len];
    //     msg.buffer[msg.len-1] = '\0';
    //     for(int i = 0; i<msg.len-1 ; ++i) {
    //         msg.buffer[i] = msg_str[i];
    //     }
    //     std::cout << "BE : " << msg.buffer << std::endl;
    //     send(fd_remote , &msg , sizeof(msg) , 0);
    //     send(fd_remote , msg.buffer ,msg.len , 0);
    //     if(tick == 200) {
    //         return;
    //     }
    // }
    // For test tick
    /////////////////////////////////////////////////////////

    std::string in;

    while (std::cin >> in) {
        std::unique_lock<std::mutex> locker(io_mutex);

        Msg msg;
        msg.tag = 0;
        msg.len = in.size() + 1;
        msg.buffer = new char[msg.len];
        msg.buffer[msg.len - 1] = '\0';

        for (int i = 0; i < msg.len - 1; ++i) {
            msg.buffer[i] = in[i];
        }

        std::cout << "BE : " << msg.buffer << std::endl;

        if (-1 == send(fd_remote, &msg, sizeof(msg), 0)) {
            continue;
        }

        if (-1 == send(fd_remote, msg.buffer, msg.len, 0)) {
            continue;
        }
    }
}

int handle_command(int fd_remote, Msg& msg) {
    std::unique_lock<std::mutex> locker(io_mutex);

    std::cout << "recive message : " << msg.tag << "\n";

    if (0 == msg.tag && msg.len > 0) {
        std::cout << "FE : " << msg.buffer << std::endl;
        return 0;
    } else if (1 == msg.tag && msg.len > 0) {
        return 0;
    } else if (-1 == msg.tag) {
        return -1;
    } else {
        std::cout << "handle nothing.\n";
        return 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "invalid input\n";
        return -1;
    }

    std::cout << "path is " << argv[1] << std::endl;

    int fd_s = socket(AF_UNIX, SOCK_STREAM, 0);

    if (fd_s == -1) {
        std::cout << "create socket failed!\n";
        return -1;
    }

    struct sockaddr_un remote;

    remote.sun_family = AF_UNIX;

    strcpy(remote.sun_path, argv[1]);

    socklen_t len = sizeof(remote);

    /////////////////////////////////////////////////////////
    //这里连了20次，每一秒连一次
    int connect_status = -1;

    for (int i = 0; i < 20; ++i) {
        connect_status = connect(fd_s, (struct sockaddr*)(&remote), len);

        if (connect_status != -1) {
            break;
        }

        std::cout << "connecting : " << i << std::endl;
        sleep(1);
    }

    /////////////////////////////////////////////////////////

    if (connect_status == -1) {
        std::cout << "connect failed!\n";
        return -1;
    }

    std::thread th(message_queue, fd_s);
    th.detach();

    std::cout << "<><>be start<><>\n";

    for (;;) {

        Msg msg;

        if (-1 == recv(fd_s, &msg, sizeof(msg), 0)) {
            std::cout << "warning read failed!\n";
            continue;
        }

        std::cout << "receive something!\n";
        std::cout << msg.len << " " << msg.tag << " \n";

        int len = msg.len;

        if (len <= 0) {
            std::cout << "buffer length is less than 0";
        } else {
            msg.buffer = new char[len];

            if (-1 == recv(fd_s, msg.buffer, len, 0)) {
                std::cout << "warning read failed!\n";
                continue;
            }
        }

        if (-1 == handle_command(fd_s, msg)) {
            break;
        }
    }

    std::cout << "<><>be quit<><>\n";

    return 0;
}