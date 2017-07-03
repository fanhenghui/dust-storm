#ifndef MED_IMG_RENDER_ALGORITHM_H
#define MED_IMG_RENDER_ALGORITHM_H

#include "med_img_pkg_config.h"

#include "MedImgUtil/mi_exception.h"

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "MedImgArithmetic/mi_arithmetic_utils.h"
#include "MedImgArithmetic/mi_vector3.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_quat4.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_vector4f.h"
#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_matrix4f.h"
#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_track_ball.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgGLResource/mi_gl_buffer.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"


MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
    #ifdef MEDIMGRENDERALGORITHM_EXPORTS
    #define RenderAlgo_Export __declspec(dllexport)
    #else
    #define RenderAlgo_Export __declspec(dllimport)
    #endif
#else
    #define RenderAlgo_Export 
#endif

#pragma warning(disable: 4251 4819 4616)

#ifndef RENDERALGO_THROW_EXCEPTION
#define RENDERALGO_THROW_EXCEPTION(desc) THROW_EXCEPTION("RenderALgorithm" , desc);
#endif

#ifndef RENDERALGO_CHECK_NULL_EXCEPTION
#define  RENDERALGO_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    RENDERALGO_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMG_END_NAMESPACE

#endif