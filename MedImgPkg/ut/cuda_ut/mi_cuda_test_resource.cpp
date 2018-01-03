#include <GL/glew.h>


#include "cudaresource/mi_cuda_texture_1d.h"
#include "cudaresource/mi_cuda_texture_2d.h"
#include "cudaresource/mi_cuda_texture_3d.h"
#include "cudaresource/mi_cuda_gl_texture_2d.h"
#include "cudaresource/mi_cuda_device_memory.h"
#include <memory>

using namespace medical_imaging;

int cuda_test_resource(int argc, char* argv[]) {
    UIDType uid = 0;
    std::shared_ptr<CudaTexture3D>  cuda_tex_3d(new CudaTexture3D(uid));
    
    return 0;
}