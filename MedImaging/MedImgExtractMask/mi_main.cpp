#include <string>

extern int ExtractMask(int argc , char* argv[]);
extern int ExtractMaskVis(int argc , char* argv[]);

int main(int argc , char* argv[])
{
    bool vis = false;
    for (int i = 1; i< argc ; ++i)
    {
        if (std::string(argv[i]) == "-vis")
        {
            vis = true;
        }
    }
    if (vis)
    {
        return ExtractMaskVis(argc , argv);
    }
    else
    {
        return ExtractMask(argc , argv);
    }
}