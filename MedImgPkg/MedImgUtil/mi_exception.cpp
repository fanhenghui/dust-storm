#ifndef MED_IMG_UTIL_EXCEPTION_H
#define MED_IMG_UTIL_EXCEPTION_H

#include "mi_exception.h"

MED_IMG_BEGIN_NAMESPACE

Exception::Exception(const std::string &module, const std::string& file, long line, const std::string& function, const std::string& description) :std::exception()
    , _module(module)
    , _line(line)
    , _function(function)
    , _file(file)
    , _description(description)
{

}

Exception::Exception(const Exception& e) :std::exception(e)
    , _module(e._module)
    , _line(e._line)
    , _function(e._function)
    , _file(e._file)
    , _description(e._description)
{

}

Exception::~Exception()
{

}

Exception& Exception::operator=(const Exception& e)
{
    _module = e._module;
    _line = e._line;
    _function = e._function;
    _file = e._file;
    _description = e._description;
    std::exception::operator=(e);
    return *this;
}

const char* Exception::what() const
{
    return get_full_description().c_str();
}

inline long Exception::get_line() const
{
    return _line;
}

const std::string& Exception::get_function() const
{
    return _function;
}

const std::string& Exception::get_file() const
{
    return _file;
}

const std::string& Exception::get_description() const
{
    return _description;
}

const std::string& Exception::get_full_description() const
{
    if (_full_description.empty()){
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


MED_IMG_END_NAMESPACE
#endif