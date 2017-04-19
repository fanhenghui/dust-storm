#include "mi_gl_program_manager.h"

MED_IMAGING_BEGIN_NAMESPACE

    GLProgramManager* GLProgramManager::m_instance = nullptr;

boost::mutex GLProgramManager::m_mutex;

GLProgramManager* GLProgramManager::Instance()
{
    if (nullptr == m_instance)
    {
        boost::unique_lock<boost::mutex> locker(m_mutex);
        if (nullptr == m_instance)
        {
            m_instance = new GLProgramManager();
        }
    }
    return m_instance;
}

GLProgramManager::GLProgramManager() :m_pUIDGen(new GLUIDGenerator())
{

}

GLProgramManager::~GLProgramManager()
{

}

GLProgramPtr GLProgramManager::CreateObject(UIDType &uid)
{
    boost::unique_lock<boost::mutex> locker(m_mutexRes);
    uid = m_pUIDGen->Tick();
    if (m_Programs.find(uid) != m_Programs.end())
    {
        GLRESOURCE_THROW_EXCEPTION("Generated UID invalid!");
    }

    GLProgramPtr pPorgram(new GLProgram(uid));
    m_Programs[uid] = pPorgram;

    return pPorgram;
}

GLProgramPtr GLProgramManager::GetObject(UIDType uid)
{
    boost::unique_lock<boost::mutex> locker(m_mutexRes);
    auto it = m_Programs.find(uid);
    if (it == m_Programs.end())
    {
        return nullptr;
    }
    else
    {
        return it->second;
    }
}

void GLProgramManager::RemoveObject(UIDType uid)
{
    boost::unique_lock<boost::mutex> locker(m_mutexRes);
    auto it = m_Programs.find(uid);
    if (it != m_Programs.end())
    {
        m_Discard.push_back(it->second);
        m_Programs.erase(it);
    }
}

void GLProgramManager::RemoveAll()
{
    boost::unique_lock<boost::mutex> locker(m_mutexRes);
    for (auto it = m_Programs.begin() ; it != m_Programs.end() ; ++it)
    {
        m_Discard.push_back(it->second);
    }
    m_Programs.clear();
}

void GLProgramManager::Update()
{
    boost::unique_lock<boost::mutex> locker(m_mutexRes);
    for (auto it = m_Discard.begin() ; it != m_Discard.end() ; )
    {
        if ((*it).use_count() > 1)
        {
            std::cout << "Cant discard useless GL program : " << (*it)->GetUID() << std::endl;
            ++it;
        }
        else
        {
            (*it)->Finalize();
            it = m_Discard.erase(it);
        }
    }
}



MED_IMAGING_END_NAMESPACE
