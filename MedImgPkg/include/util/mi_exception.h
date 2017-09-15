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
    Exception(const std::string& module, const std::string& file, long line,
        const std::string& function, const std::string& description) : std::exception()
        , _module(module)
        , _line(line)
        , _function(function)
        , _file(file)
        , _description(description) 
    {}

    Exception(const Exception& e): std::exception(e)
        , _module(e._module)
        , _line(e._line)
        , _function(e._function)
        , _file(e._file)
        , _description(e._description) 
    {}

    virtual ~Exception() {}

    Exception& operator=(const Exception& e) {
        _module = e._module;
        _line = e._line;
        _function = e._function;
        _file = e._file;
        _description = e._description;
        std::exception::operator=(e);
        return *this;
    }

    virtual const char* what() const throw() {
        return get_full_description().c_str();
    }

    long get_line() const {
        return _line;
    }

    const std::string& get_function() const {
        return _function;
    }

    const std::string& get_file() const {
        return _file;
    }

    const std::string& get_description() const {
        return _description;
    }

    const std::string& get_full_description() const {
        if (_full_description.empty()) {
            std::stringstream ss;
    
            ss << _module << " Exception<"
               << " File : " << _file << " ,"
               << " Line : " << _line << " ,"
               << " Function : " << _function << " ,"
               << " Description : " << _description
               << " >";
    
            _full_description = ss.str();
        }
    
        return _full_description;
    }

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

#ifndef UTIL_THROW_EXCEPTION
#define UTIL_THROW_EXCEPTION(desc) THROW_EXCEPTION("Util" , desc);
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
