#include "mi_meta_object_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"

MED_IMAGING_BEGIN_NAMESPACE


IOStatus MetaObjectLoader::load(
const std::string& info_file, 
std::shared_ptr<ImageData> &image_data , 
std::shared_ptr<MetaObjectTag> & meta_obj_tag,
std::shared_ptr<ImageDataHeader> & img_data_header)
{
    //Get header
    meta_obj_tag.reset(new MetaObjectTag());

    std::ifstream in(info_file.c_str() , std::ios::in);
    if (!in.is_open())
    {
        IO_THROW_EXCEPTION("Cant open meta file!");
    }

    std::string sLine;
    std::string tag;
    std::string sEqual;
    std::string context;
    while(std::getline(in , sLine))
    {
        std::stringstream ss(sLine);
        ss >> tag;
        if (tag == "ObjectType")
        {
            ss >> sEqual >> context;
            if (context.empty())
            {
                meta_obj_tag->object_type = context;
            }
        }
        else if (tag == "ElementSpacing")
        {
            ss >> sEqual >> meta_obj_tag->spacing[0] >> meta_obj_tag->spacing[1] >> meta_obj_tag->spacing[2];
        }
        else if (tag == "DimSize")
        {
            ss >> sEqual>>meta_obj_tag->dim_size[0] >>meta_obj_tag->dim_size[1]>>meta_obj_tag->dim_size[2];
        }

    }

    in.close();

    return IO_SUCCESS;
}

MED_IMAGING_END_NAMESPACE