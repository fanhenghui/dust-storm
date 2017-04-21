#ifndef MED_IMAGING_COMMON_EXCEPTION_H_
#define MED_IMAGING_COMMON_EXCEPTION_H_

#include "MedImgCommon/mi_common_stdafx.h"

#include <exception>
#include <string>
#include <sstream>

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export Exception : public std::exception
{
public:
    Exception(const std::string &module, const std::string& file, long line, const std::string& func, const std::string& des);

    Exception(const Exception& e);

    virtual ~Exception();

    Exception& operator=(const Exception& e);

    const char* what() const;

    long get_line() const;

    const std::string& get_function() const;

    const std::string& get_file() const;

    const std::string& get_description() const;

    const std::string& get_full_description() const;

protected:
private:
    std::string _module;
    long _line;
    std::string _function;
    std::string _file;
    std::string _description;
    mutable std::string _full_description;
};




#ifndef THROW_EXCEPTION
#define THROW_EXCEPTION(module , desc) throw Exception(module  , __FILE__ , __LINE__ , __FUNCTION__ , desc);
#endif

#ifndef COMMON_THROW_EXCEPTION
#define COMMON_THROW_EXCEPTION(desc) THROW_EXCEPTION("Common" , desc);
#endif

#ifndef COMMON_CHECK_NULL_EXCEPTION
#define  COMMON_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    COMMON_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMAGING_END_NAMESPACE

#endif
