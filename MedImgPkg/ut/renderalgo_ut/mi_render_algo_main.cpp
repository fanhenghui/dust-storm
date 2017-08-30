#include <iostream>

extern int TE_MPRScene(int argc, char* argv[]);
extern int TE_VRScene(int argc, char* argv[]);
int main(int argc, char* argv[]) {
    return TE_VRScene(argc, argv);
}