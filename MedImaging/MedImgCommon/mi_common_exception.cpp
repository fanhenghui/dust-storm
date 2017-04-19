#ifndef MED_IMAGING_COMMON_EXCEPTION_H
#define MED_IMAGING_COMMON_EXCEPTION_H

#include "mi_common_exception.h"

MED_IMAGING_BEGIN_NAMESPACE

Exception::Exception(const std::string &sModule, const std::string& sFile, long iLine, const std::string& sFunction, const std::string& sDes) :std::exception()
    , m_sModule(sModule)
    , m_iLine(iLine)
    , m_sFunction(sFunction)
    , m_sFile(sFile)
    , m_sDescription(sDes)
{

}

Exception::Exception(const Exception& e) :std::exception(e)
    , m_sModule(e.m_sModule)
    , m_iLine(e.m_iLine)
    , m_sFunction(e.m_sFunction)
    , m_sFile(e.m_sFile)
    , m_sDescription(e.m_sDescription)
{

}

Exception::~Exception()
{

}

Exception& Exception::operator=(const Exception& e)
{
    m_sModule = e.m_sModule;
    m_iLine = e.m_iLine;
    m_sFunction = e.m_sFunction;
    m_sFile = e.m_sFile;
    m_sDescription = e.m_sDescription;
    std::exception::operator=(e);
    return *this;
}

const char* Exception::what() const
{
    return GetFullDescription().c_str();
}

inline long Exception::GetLine() const
{
    return m_iLine;
}

const std::string& Exception::GetFunction() const
{
    return m_sFunction;
}

const std::string& Exception::GetFile() const
{
    return m_sFile;
}

const std::string& Exception::GetDescription() const
{
    return m_sDescription;
}

const std::string& Exception::GetFullDescription() const
{
    if (m_sFullDescription.empty())
    {
        std::stringstream ss;

        ss << m_sModule << " Exception<"
            << " File : " << m_sFile << " ,"
            << " Line : " << m_iLine << " ,"
            << " Function : " << m_sFunction << " ,"
            << " Description : " << m_sDescription
            << " >";

        m_sFullDescription = ss.str();
    }

    return m_sFullDescription;
}


MED_IMAGING_END_NAMESPACE
#endif