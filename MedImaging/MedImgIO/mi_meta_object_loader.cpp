#include "mi_meta_object_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"

MED_IMAGING_BEGIN_NAMESPACE


IOStatus MetaObjectLoader::load(
const std::string& sInfoFile, 
std::shared_ptr<ImageData> &pImgData , 
std::shared_ptr<MetaObjectTag> & pMetaObjTag,
std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    //Get header
    pMetaObjTag.reset(new MetaObjectTag());

    std::ifstream in(sInfoFile.c_str() , std::ios::in);
    if (!in.is_open())
    {
        IO_THROW_EXCEPTION("Cant open meta file!");
    }

    std::string sLine;
    std::string sTag;
    std::string sEqual;
    std::string sContent;
    while(std::getline(in , sLine))
    {
        std::stringstream ss(sLine);
        ss >> sTag;
        if (sTag == "ObjectType")
        {
            ss >> sEqual >> sContent;
            if (sContent.empty())
            {
                pMetaObjTag->m_sObjectType = sContent;
            }
        }
        else if (sTag == "ElementSpacing")
        {
            ss >> sEqual >> pMetaObjTag->m_dSpacing[0] >> pMetaObjTag->m_dSpacing[1] >> pMetaObjTag->m_dSpacing[2];
        }
        else if (sTag == "DimSize")
        {
            ss >> sEqual>>pMetaObjTag->m_uiDimSize[0] >>pMetaObjTag->m_uiDimSize[1]>>pMetaObjTag->m_uiDimSize[2];
        }

    }

    in.close();

    return IO_SUCCESS;
}

MED_IMAGING_END_NAMESPACE