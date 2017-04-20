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
    Exception(const std::string &sModule, const std::string& sFile, long iLine, const std::string& sFunction, const std::string& sDes);

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
    std::string m_sModule;
    long m_iLine;
    std::string m_sFunction;
    std::string m_sFile;
    std::string m_sDescription;
    mutable std::string m_sFullDescription;
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
