#include <stdio.h>
#include <stdlib.h>
#include <WinSock2.h>
#include <iostream>
#include <string>
#include <fstream>
#include "mbedtls/md5.h"

namespace
{
    std::string img_file = "D:/temp/AB_CTA_01.raw";

    void get_img_hash(unsigned char (&output)[16])
    {
        memset(output, 0, 16);
        std::ifstream in(img_file, std::ios::in | std::ios::binary);
        if (in.is_open())
        {
            unsigned char* img_data = new unsigned char[1024 * 1024 * 3];
            in.read((char*)img_data, 1024 * 1024 * 3);

            
            mbedtls_md5(img_data, 1024 * 1024 * 3, output);
            in.close();
        }
    }
    
}

int main() 
{
    //初始化DLL
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    //创建套接字
    SOCKET sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    //向服务器发起请求
    sockaddr_in sockAddr;
    memset(&sockAddr, 0, sizeof(sockAddr));  //每个字节都用0填充
    sockAddr.sin_family = PF_INET;
    sockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    sockAddr.sin_port = htons(1234);
    connect(sock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));

    //向服务器发送数据
    unsigned char md5[16] = {0};
    get_img_hash(md5);
    send(sock, (char*)(&(md5[0])), 16, NULL);


    //接收服务器传回的数据
    //float sum;
    //recv(sock, (char*)(&sum), sizeof(float), NULL);
    ////输出接收到的数据
    //std::cout << "Sum result : " << sum << std::endl;
    //关闭套接字
    closesocket(sock);

    return 0;
}