#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>


namespace
{
    static const std::string sTagEcoderDst = "EncoderDst";
    static const std::string sTagEcoderItem = "EncoderItem";
}
void main(int argc , char* argv[])
{
    if (argc < 2)
    {
        std::cout <<"Parameter valid!\n";
        return;
    }

    //1 Parse config file
    const char* pConfigFile = argv[1];
    std::fstream inConfig(pConfigFile , std::ios::in);
    if (!inConfig.is_open())
    {
        std::cout <<"Cant open config file " << inConfig << std::endl;
        return;
    }

    std::string sLine;
    std::ofstream outEncodDstFile;
    bool bDstFileValid = false;
    while(std::getline(inConfig , sLine))
    {
        if (sLine.empty())
        {
            continue;
        }

        std::stringstream ss(sLine);
        std::string sTag;
        std::string sEqualSymbol;
        std::string sContext0;
        std::string sContext1;
        ss >> sTag ;
        if (sTag == sTagEcoderDst)//Begin a encoding to file sContext0
        {
            ss >> sEqualSymbol >> sContext0;
            outEncodDstFile.clear();
            outEncodDstFile.close();
            bDstFileValid = false;
            outEncodDstFile.open(sContext0 , std::ios::out);
            if (!outEncodDstFile.is_open())
            {
                std::cout << "Cant open encoding dst file : " << sContext0<< std::endl;
                continue;
            }
            //Write header
            outEncodDstFile << "#pragma  once\n\n";
            bDstFileValid = true;
        }
        else if (sTag == sTagEcoderItem)//Encoder shader file sContext0 to opened dst file
        {
            ss >> sEqualSymbol >> sContext0 >> sContext1;
            if (!bDstFileValid)
            {
                continue;
            }

            //Read shader file and write to dst file
            if (sContext1.empty() || sContext0.empty())
            {
                continue;
            }

            std::ifstream inShader(sContext0.c_str() , std::ios::in);
            if (!inShader.is_open())
            {
                std::cout << "Cant open shader file " << sContext0 << std::endl;
                continue;
            }

            //Write item variable
            outEncodDstFile << "static const char* " << sContext1 <<" = \"\\\n";

            std::string sShaderLine;
            while(std::getline(inShader, sShaderLine))
            {
                outEncodDstFile << sShaderLine <<"\\n" << "\\\n";
            }
            //Write item end;
            outEncodDstFile << "\";\n\n";
            inShader.close();
        }
        else//print as annotation
        {
            outEncodDstFile <<"//" << sLine << std::endl;
        }
    }
    outEncodDstFile.clear();
    outEncodDstFile.close();
    inConfig.close();
}