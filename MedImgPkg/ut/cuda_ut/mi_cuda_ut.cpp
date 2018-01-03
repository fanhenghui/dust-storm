#include <stdio.h>
extern int cuda_texture(int argc, char *argv[]);
extern int mi_cuda_vr(int argc, char* argv[]);
extern int mi_simple_ray_tracing(int argc, char* argv[]);
extern int cuda_test_resource(int argc, char* argv[]);

int main(int argc, char* argv[]) 
{
    cuda_test_resource(argc,  argv);
}