#include <stdio.h>
extern int cuda_texture(int argc, char *argv[]);
extern int mi_cuda_vr(int argc, char* argv[]);
int main(int argc, char* argv[]) 
{
    mi_cuda_vr(argc,  argv);
}