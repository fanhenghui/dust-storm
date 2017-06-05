#include <stdlib.h>
#include <stdio.h>
#include <winsock2.h>
#include <iostream>
#include <string>
#include <sstream>

//mysql begin
#include "mysql_connection.h"
#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/resultset.h"
#include "cppconn/statement.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/sqlstring.h"
//mysql end

//#pragma comment (lib, "ws2_32.lib")  //加载 ws2_32.dll

/*单个字符转十六进制字符串，长度增大2被*/
static void Char2Hex(unsigned char ch, char *szHex)
{
    int i;
    unsigned char byte[2];
    byte[0] = ch / 16;
    byte[1] = ch % 16;
    for (i = 0; i < 2; i++)
    {
        if (byte[i] >= 0 && byte[i] <= 9)
            szHex[i] = '0' + byte[i];
        else
            szHex[i] = 'a' + byte[i] - 10;
    }
    szHex[2] = 0;
}

/*字符串转换函数，中间调用上面的函数*/
void CharStr2HexStr(char *pucCharStr, int iSize, char *pszHexStr)
{
    int i;
    char szHex[3];
    pszHexStr[0] = 0;
    for (i = 0; i < iSize; i++)
    {
        Char2Hex(pucCharStr[i], szHex);
        strcat(pszHexStr, szHex);
    }
}

//https://dev.mysql.com/doc/connector-cpp/en/connector-cpp-examples-complete-example-1.html
std::string query_path(char md5hex[32])
{
    try
    {
        sql::Driver *driver = nullptr;
        sql::Connection *con = nullptr;
        sql::Statement *stmt = nullptr;
        sql::PreparedStatement *pstmt = nullptr;
        sql::ResultSet *res = nullptr;

        //create connect
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "0123456");
        con->setSchema("pathology");

        stmt = con->createStatement();
        delete stmt;

        std::stringstream ss;
        ss << "SELECT * FROM images WHERE md5=\'";
        for (int i= 0 ; i<32 ; ++i)
        {
            ss << md5hex[i];
        }
        ss << "\';";
        //std::cout << "query cmd : " << ss.str().c_str() << std::endl;

        pstmt = con->prepareStatement(ss.str().c_str());
        res = pstmt->executeQuery();

        res->afterLast();
        while (res->previous())
        {
            std::string path = res->getString("path");

            delete res; 
            res = nullptr;
            delete pstmt;
            pstmt = nullptr;
            delete con;
            con = nullptr;

            //std::cout << "\npath : " <<path << std::endl ;
            return path;
        }

        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
        delete con;
        con = nullptr;

        return std::string("");
    }
    catch (const sql::SQLException& e)
    {
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    }
}


int main()
{
    /*std::string md5_str = "6c9210566cb1fd1c5b5f493fbd0f1388";
    char md5_hex[32];
    for (int i = 0; i < 32; ++i)
    {
        md5_hex[i] = md5_str[i];
    }
    query_path(md5_hex);
    return 0;*/

    //初始化 DLL
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    //创建套接字
    SOCKET servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    //绑定套接字
    sockaddr_in sockAddr;
    memset(&sockAddr, 0, sizeof(sockAddr));  //每个字节都用0填充
    sockAddr.sin_family = PF_INET;  //使用IPv4地址
    sockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");  //具体的IP地址
    sockAddr.sin_port = htons(1234);  //端口
    bind(servSock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));

    //进入监听状态
    listen(servSock, 20);

    while (1)
    {
        //接收客户端请求
        SOCKADDR clntAddr;
        int nSize = sizeof(SOCKADDR);
        SOCKET clntSock = accept(servSock, (SOCKADDR*)&clntAddr, &nSize);

        char md5[16];
        recv(clntSock, (char*)(md5), 16, NULL);
        int *md5_int = (int*)md5;
        char md5_str[16 * 2];
        CharStr2HexStr(md5, 16, md5_str);
        std::cout << "calculate md5 is :" << md5_str << "\t";
        for (int i = 0 ;i<4 ; ++i)
        {
            std::cout << md5_int[i];
        }
        std::cout <<  std::endl;

        //查询数据库
        std::string path =  query_path(md5_str);
        if (!path.empty())
        {
            std::cout << "Get data path : " << path << std::endl;
        }
        ////向客户端发送数据
        //send(clntSock, (char*)(&sum), sizeof(float), NULL);

        //std::cout << "calculate result : " << sum << std::endl;

        //关闭套接字
        closesocket(clntSock);

    }

    closesocket(servSock);
    //终止 DLL 的使用
   WSACleanup();
    return 0;
}