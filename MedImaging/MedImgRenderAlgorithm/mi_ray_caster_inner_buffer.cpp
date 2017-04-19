#include "mi_ray_caster_inner_buffer.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_buffer.h"

MED_IMAGING_BEGIN_NAMESPACE

struct RayCasterInnerBuffer::GLResource
{
    std::map<BufferType , GLBufferPtr> m_mapBufferID;
    bool m_DirtyFlag[TypeEnd];

    GLResource()
    {
        memset(m_DirtyFlag , 1 , sizeof(bool)*TypeEnd);
    }

    void Release()
    {
        for (auto it = m_mapBufferID.begin() ; it != m_mapBufferID.end() ; ++it)
        {
            GLResourceManagerContainer::Instance()->GetBufferManager()->RemoveObject(it->second->GetUID());
        }
        m_mapBufferID.clear();
        memset(m_DirtyFlag , 0 , sizeof(bool)*TypeEnd);
    }

    GLBufferPtr GetBuffer(BufferType eType)
    {
        auto it = m_mapBufferID.find(eType);
        if (it != m_mapBufferID.end())
        {
            return it->second;
        }
        else
        {
            UIDType uidBuffer= 0;
            GLBufferPtr pBuffer = GLResourceManagerContainer::Instance()->GetBufferManager()->CreateObject(uidBuffer);
            pBuffer->Initialize();
            pBuffer->SetBufferTarget(GL_SHADER_STORAGE_BUFFER);
            m_mapBufferID[eType] = pBuffer;
            return pBuffer;
        }
    }
};

RayCasterInnerBuffer::RayCasterInnerBuffer():m_pRes(new GLResource())
{

}

RayCasterInnerBuffer::~RayCasterInnerBuffer()
{
    ReleaseBuffer();
}

void RayCasterInnerBuffer::ReleaseBuffer()
{
    m_pRes->Release();
}

GLBufferPtr RayCasterInnerBuffer::GetBuffer(BufferType eType)
{
    try
    {
        GLBufferPtr pBuffer = m_pRes->GetBuffer(eType);

        CHECK_GL_ERROR;

        switch(eType)
        {
        case WindowLevelBucket:
            {
                if (m_pRes->m_DirtyFlag[WindowLevelBucket])
                {
                    float* pWL = (float*)m_pSharedBufferArray.get();
                    memset(pWL , 0 , sizeof(float)*static_cast<int>(m_eLabelLevel)*2);

                    for (auto it = m_mapWindowLevel.begin() ; it != m_mapWindowLevel.end() ; ++it)
                    {
                        const unsigned char uLabel = it->first;
                        if (uLabel > static_cast<int>(m_eLabelLevel) - 1)
                        {
                            std::stringstream ss;
                            ss << "Input window level label : " << (int)(uLabel) << " is greater than the limit : " << static_cast<int>(m_eLabelLevel) - 1 << " !";
                            RENDERALGO_THROW_EXCEPTION(ss.str());
                        }
                        pWL[uLabel*2] = it->second.m_Value.x;
                        pWL[uLabel*2+1] = it->second.m_Value.y;
                    }

                    pBuffer->Bind();
                    pBuffer->Load(static_cast<int>(m_eLabelLevel)*sizeof(float)*2 , pWL, GL_DYNAMIC_DRAW);

                    m_pRes->m_DirtyFlag[WindowLevelBucket] = false;
                }
                break;
            }

        case VisibleLabelBucket:
            {
                if (m_pRes->m_DirtyFlag[VisibleLabelBucket])
                {
                    int* pLabel = (int*)m_pSharedBufferArray.get();
                    memset(pLabel , 0 , sizeof(int)*static_cast<int>(m_eLabelLevel));

                    for (auto it = m_vecVisibleLabel.begin() ; it != m_vecVisibleLabel.end() ; ++it)
                    {
                        if (*it > static_cast<int>(m_eLabelLevel) - 1)
                        {
                            std::stringstream ss;
                            ss << "Input visible label : " << (int)(*it ) << " is greater than the limit : " << static_cast<int>(m_eLabelLevel) - 1 << " !";
                            RENDERALGO_THROW_EXCEPTION(ss.str());
                        }
                        pLabel[*it] = 1;
                    }

                    pBuffer->Bind();
                    pBuffer->Load(static_cast<int>(m_eLabelLevel)*sizeof(int) , pLabel , GL_STATIC_DRAW);

                    m_pRes->m_DirtyFlag[VisibleLabelBucket] = false;
                }
                break;
            }

        default:
            {
                RENDERALGO_THROW_EXCEPTION("Invalid buffer type!");
            }
        }

        CHECK_GL_ERROR;

        return pBuffer;

    }
    catch (Exception& e)
    {
#ifdef _DEBUG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
}

void RayCasterInnerBuffer::SetMaskLabelLevel(LabelLevel eLevel)
{
    if (m_eLabelLevel != eLevel)
    {
        m_eLabelLevel = eLevel;
        memset(m_pRes->m_DirtyFlag , 1 , sizeof(bool)*TypeEnd);
        m_pSharedBufferArray.reset(new char[static_cast<int>(m_eLabelLevel)*2*4]);
    }
}

void RayCasterInnerBuffer::SetWindowLevel(float fWW , float fWL , unsigned char ucLabel)
{
    const Vector2f vWL(fWW , fWL);
    auto it = m_mapWindowLevel.find(ucLabel);
    if (it == m_mapWindowLevel.end())
    {
        m_mapWindowLevel.insert(std::make_pair(ucLabel ,vWL));
        m_pRes->m_DirtyFlag[WindowLevelBucket] = true;
    }
    else
    {
        if (vWL != it->second)
        {
            it->second = vWL;
            m_pRes->m_DirtyFlag[WindowLevelBucket] = true;
        }
    }
}

void RayCasterInnerBuffer::SetVisibleLabels(std::vector<unsigned char> vecLabels)
{
    if (m_vecVisibleLabel != vecLabels)
    {
        m_vecVisibleLabel =vecLabels;
        m_pRes->m_DirtyFlag[VisibleLabelBucket] = true;
    }
}

MED_IMAGING_END_NAMESPACE