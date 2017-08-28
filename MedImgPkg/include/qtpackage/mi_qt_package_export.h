#ifndef MED_IMG_QT_WIDGETS_H
#define MED_IMG_QT_WIDGETS_H

#include "med_img_pkg_config.h"
#include "gl/glew.h"

#include <QtCore/qglobal.h>

#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "util/mi_exception.h"


#ifdef MEDIMGQTPACKAGE_LIB
# define QtPackage_Export Q_DECL_EXPORT
#else
# define QtPackage_Export Q_DECL_IMPORT
#endif

#pragma warning(disable: 4251 4819)

#ifndef QTWIDGETS_THROW_EXCEPTION
#define QTWIDGETS_THROW_EXCEPTION(desc) throw medical_imaging::Exception("QtWidgets"  , __FILE__ , __LINE__ , __FUNCTION__ , desc);
#endif

#ifndef QTWIDGETS_CHECK_NULL_EXCEPTION
#define  QTWIDGETS_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    QTWIDGETS_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

#endif