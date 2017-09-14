#ifndef MED_IMAGING_QT_WIDGETS_H
#define MED_IMAGING_QT_WIDGETS_H

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
#include "util/mi_logger.h"


#ifdef MEDIMGQTPACKAGE_LIB
# define QtWidgets_Export Q_DECL_EXPORT
#else
# define QtWidgets_Export Q_DECL_IMPORT
#endif

#pragma warning(disable: 4251 4819)

#ifndef QTWIDGETS_THROW_EXCEPTION
#define QTWIDGETS_THROW_EXCEPTION(desc) throw medical_imaging::Exception("QtPackage"  , __FILE__ , __LINE__ , __FUNCTION__ , desc);
#endif

#ifndef QTWIDGETS_CHECK_NULL_EXCEPTION
#define  QTWIDGETS_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    QTWIDGETS_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

src::severity_logger<medical_imaging::SeverityLevel> G_QT_PACKAGE_LG;
#define MI_QT_PACKAGE_LOG(sev) BOOST_LOG_SEV(G_QT_PACKAGE_LG, sev)

#endif