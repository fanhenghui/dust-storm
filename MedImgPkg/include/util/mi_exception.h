#ifndef MEDIMGUTIL_MI_EXCEPTION_H
#define MEDIMGUTIL_MI_EXCEPTION_H

#include "util/mi_util_export.h"

#include <exception>
#include <string>
#include <sstream>
#include <typeinfo>

MED_IMG_BEGIN_NAMESPACE

class Exception : public std::exception {
public:
    Exception(const std::string& module, const std::string& file, long line, const std::string& func,
              const std::string& des);

    Exception(const Exception& e);

    virtual ~Exception();

    Exception& operator=(const Exception& e);

    virtual const char* what() const throw();

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

#include "mi_exception.inl"


#ifndef THROW_EXCEPTION
#define THROW_EXCEPTION(module , desc) throw Exception(module  , __FILE__ , __LINE__ , __FUNCTION__ , desc);
#endif

#ifndef UTIL_THROW_EXCEPTION
#define UTIL_THROW_EXCEPTION(desc) THROW_EXCEPTION("Common" , desc);
#endif

#ifndef  UTIL_CHECK_NULL_EXCEPTION
#define   UTIL_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    UTIL_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMG_END_NAMESPACE

#endif
