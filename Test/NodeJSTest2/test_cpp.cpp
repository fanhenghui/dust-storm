#include <iostream>
#include <string>

int main(int argc , char* argv[])
{
    std::cout << "test cpp>>>>>>\n";
    std::cout << "running>>>>>>\n";
    std::string in;
    while(true)
    {
        std::cin >> in;
        if(in == "exit")
        {
            break;
        }
    }   
}