#include "mi_meta_object_loader.h"

#include "util/mi_string_number_converter.h"

#include "mi_image_data.h"
#include "mi_image_data_header.h"
#include "mi_zlib_utils.h"

MED_IMG_BEGIN_NAMESPACE

IOStatus MetaObjectLoader::load(
const std::string& info_file, 
std::shared_ptr<ImageData> &image_data , 
std::shared_ptr<MetaObjectTag> & meta_obj_tag,
std::shared_ptr<ImageDataHeader> & img_data_header)
{
    //1 construct meta obj tag
    IOStatus status = construct_meta_object_tag_i(info_file , meta_obj_tag);
    if( status != IO_SUCCESS)
    {
        return status;
    }

    //2 construct data header
    status = construct_data_header_i(meta_obj_tag , img_data_header , image_data);
    if( status != IO_SUCCESS)
    {
        return status;
    }

    //3 construct image data
    status = construct_image_data_i(info_file , meta_obj_tag , img_data_header , image_data);
    if( status != IO_SUCCESS)
    {
        return status;
    }

    return IO_SUCCESS;
}

IOStatus MetaObjectLoader::construct_meta_object_tag_i( const std::string& info_file , std::shared_ptr<MetaObjectTag> & meta_obj_tag )
{
    meta_obj_tag.reset(new MetaObjectTag());

    std::ifstream in(info_file.c_str() , std::ios::in);
    if (!in.is_open())
    {
        return IO_FILE_OPEN_FAILED;
    }

    std::string sLine;
    std::string tag;
    std::string sEqual;
    std::string context;
    while(std::getline(in , sLine))
    {
        std::stringstream ss(sLine);
        ss >> tag >> sEqual;
        if (tag == META_COMMENT)
        {
            ss >> context;
            meta_obj_tag->comment = context;
        }
        else if (tag == META_OBJECT_TYPE)
        {
            ss >> context;
            meta_obj_tag->object_type = context;
        }
        else if (tag == META_OBJECT_SUB_TYPE)
        {
            ss >> context;
            meta_obj_tag->object_sub_type = context;
        }
        else if (tag == META_TRANSFORM_TYPE)
        {
            ss >> context;
            meta_obj_tag->transform_type = context;
        }
        else if (tag == META_NDIMS)
        {
            int ndims(0);
            ss >> ndims;
            meta_obj_tag->n_dims= ndims;
        }
        else if (tag == META_NAME)
        {
            ss >> context;
            meta_obj_tag->name = context;
        }
        else if (tag == META_ID)
        {
            int id(0);
            ss << id;
            meta_obj_tag->id = id;
        }
        else if (tag == META_PATIENT_ID)
        {
            int patient_id(0);
            ss << patient_id;
            meta_obj_tag->parent_id = patient_id;
        }
        else if (tag == META_BINARY_DATA)
        {
            ss >> context;
            if (context == "True")
            {
                meta_obj_tag->is_binary_data = true;
            }
            else
            {
                meta_obj_tag->is_binary_data = false;
            }
        }
        else if (tag == META_ELEMENT_BYTE_ORDER_MSB)
        {
            ss >> context;
            meta_obj_tag->element_byte_order_msb= context;
        }
        else if (tag == META_BINARY_DATA_BYTE_ORDER_MSB)
        {
            ss >> context;
            meta_obj_tag->binary_data_byte_order_msb = context;
        }
        else if (tag == META_COMPRESSD_DATA)
        {
            ss >> context;
            if (context == "True")
            {
                meta_obj_tag->is_compressed_data = true;
            }
            else
            {
                meta_obj_tag->is_compressed_data = false;
            }
        }
        else if (tag == META_COMPRESSED_DATA_SIZE)
        {
            ss >> meta_obj_tag->compressed_data_size;
        }
        else if (tag == META_COLOR)
        {
            float r , g , b , a;
            ss >> r >> g >> b >> a;
            meta_obj_tag->color[0] = r;
            meta_obj_tag->color[1] = g;
            meta_obj_tag->color[2] = b;
            meta_obj_tag->color[3] = a;
        }
        else if (tag == META_POSITION || tag == META_OFFSET || tag == META_ORIGIN)
        {
            double x , y , z;
            if (3 == meta_obj_tag->n_dims)
            {
                ss >> x >> y >> z;
                meta_obj_tag->position.x = x;
                meta_obj_tag->position.y = y;
                meta_obj_tag->position.z = z;
            }
            else
            {
                return IO_UNSUPPORTED_YET;
            }
        }
        else if (tag == META_ORIENTATION || tag == META_TRANSFORM_MATRIX || tag == META_ROTATION)
        {
            double x , y , z , x1 , y1 , z1 , x2 , y2 , z2; 
            if (3 == meta_obj_tag->n_dims)
            {
                ss >> x >> y >> z >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
                meta_obj_tag->orientation_x = Vector3(x,y,z);
                meta_obj_tag->orientation_y = Vector3(x1,y1,z1);
                meta_obj_tag->orientation_z = Vector3(x2,y2,z2);
            }
            else
            {
                return IO_UNSUPPORTED_YET;
            }
        }
        else if (tag == META_ANATOMICAL_ORIENTATION)
        {
            ss >> context;
            meta_obj_tag->anatomical_orientation = context;
        }
        else if (tag == META_ELEMENT_SPACING)
        {
            ss  >> meta_obj_tag->spacing[0] >> meta_obj_tag->spacing[1] >> meta_obj_tag->spacing[2];
        }
        else if (tag == META_DIM_SIZE)
        {
            ss >>meta_obj_tag->dim_size[0] >>meta_obj_tag->dim_size[1]>>meta_obj_tag->dim_size[2];
        }
        else if (tag == META_HEADER_SIZE)
        {
            int header_size(0);
            ss >> header_size;
            meta_obj_tag->header_size = header_size;
        }
        else if (tag == META_MODALITY)
        {
            ss >> context;
            meta_obj_tag->modality = context;
        }
        else if (tag == META_SEQUENCE_ID)
        {
            int x , y , z , w;
            ss >> x >> y >> z >> w;
            meta_obj_tag->sequence_id[0] = x;
            meta_obj_tag->sequence_id[1] = y;
            meta_obj_tag->sequence_id[2] = z;
            meta_obj_tag->sequence_id[3] = w;
        }
        else if (tag == META_ELEMENT_MIN)
        {
            float ele_min(0);
            ss >> ele_min;
            meta_obj_tag->element_min = ele_min;
        }
        else if (tag == META_ELEMENT_MAX)
        {
            float ele_max(0);
            ss >> ele_max;
            meta_obj_tag->element_max = ele_max;
        }
        else if (tag == META_ELEMENT_NUMBER_OF_CHANNELS)
        {
            int channels(0);
            ss >> channels;
            meta_obj_tag->element_number_of_channels = channels;
        }
        else if (tag == META_ELEMENT_SIZE)
        {
            double x , y , z ;
            if (3 == meta_obj_tag->n_dims)
            {
                ss >> x >> y >> z;
                meta_obj_tag->element_size[0] = x;
                meta_obj_tag->element_size[1] = y;
                meta_obj_tag->element_size[2] = z;
            }
            else
            {
                return IO_UNSUPPORTED_YET;
            }
        }
        else if (tag == META_ELEMENT_TYPE)
        {
            ss >> context;
            meta_obj_tag->element_type = context;
        }
        else if (tag == META_ELEMENT_DATA_FILE)
        {
            ss >> context;
            meta_obj_tag->element_data_file= context;
        }
    }

    in.close();

    //Check meta obj tags
    if (meta_obj_tag->n_dims != 3 ||
        meta_obj_tag->dim_size[0] == 0 || 
        meta_obj_tag->dim_size[1] == 0 || 
        meta_obj_tag->dim_size[2] == 0 || 
        fabs(meta_obj_tag->spacing[0]) <  DOUBLE_EPSILON ||
        fabs(meta_obj_tag->spacing[1]) <  DOUBLE_EPSILON ||
        fabs(meta_obj_tag->spacing[2]) <  DOUBLE_EPSILON ||
        meta_obj_tag->orientation_x == Vector3::S_ZERO_VECTOR ||
        meta_obj_tag->orientation_y == Vector3::S_ZERO_VECTOR ||
        meta_obj_tag->orientation_z == Vector3::S_ZERO_VECTOR ||
        //meta_obj_tag->is_compressed_data ||
        !meta_obj_tag->is_binary_data)
    {
        return IO_UNSUPPORTED_YET;
    }

    return IO_SUCCESS;
}

IOStatus MetaObjectLoader::construct_data_header_i(
    std::shared_ptr<MetaObjectTag> meta_obj_tag , 
    std::shared_ptr<ImageDataHeader> & img_data_header ,
    std::shared_ptr<ImageData> & img_data)
{
    img_data_header.reset( new ImageDataHeader());
    img_data.reset( new ImageData());

    //1 construct image data parameter
    img_data->_channel_num = meta_obj_tag->element_number_of_channels;
    for (int i = 0 ; i<3  ; ++i)
    {
        img_data->_dim[i] = meta_obj_tag->dim_size[i];
        img_data->_spacing[i] = meta_obj_tag->spacing[i];
    }
    //data type
    if (meta_obj_tag->element_type == "MET_UCHAR")
    {
        img_data->_data_type = UCHAR;
        img_data_header->bits_allocated = 1;
        img_data_header->pixel_representation = 0;
    }
    else if (meta_obj_tag->element_type == "MET_CHAR")
    {
        img_data->_data_type = CHAR;
        img_data_header->bits_allocated = 1;
        img_data_header->pixel_representation = 1;
    }
    else if (meta_obj_tag->element_type == "MET_USHORT")
    {
        img_data->_data_type = USHORT;
        img_data_header->bits_allocated = 2;
        img_data_header->pixel_representation = 0;
    }
    else if (meta_obj_tag->element_type == "MET_SHORT")
    {
        img_data->_data_type = SHORT;
        img_data_header->bits_allocated = 2;
        img_data_header->pixel_representation = 1;
    }
    else if (meta_obj_tag->element_type == "MET_FLOAT")
    {
        img_data->_data_type = FLOAT;
        img_data_header->bits_allocated = 4;
        img_data_header->pixel_representation = 1;
    }
    else
    {
        return IO_UNSUPPORTED_YET;
    }

    img_data->_image_position = meta_obj_tag->position;
    img_data->_image_orientation[0] = meta_obj_tag->orientation_x;
    img_data->_image_orientation[1] = meta_obj_tag->orientation_y;
    img_data->_image_orientation[2] = meta_obj_tag->orientation_z;

    //2 construct image data header
    img_data_header->data_source_uid = 1;
    if (meta_obj_tag->modality == "MET_MOD_CT")
    {
        img_data_header->modality = CT;
    }
    else if (meta_obj_tag->modality == "MET_MOD_MR")
    {
        img_data_header->modality = MR;
    }
    else if (meta_obj_tag->modality == "MET_MOD_US")
    {
        img_data_header->modality = MODALITY_UNDEFINED;
    }
    else if (meta_obj_tag->modality == "MET_MOD_OTHER")
    {
        img_data_header->modality = MODALITY_UNDEFINED;
    }
    else if (meta_obj_tag->modality == "MET_MOD_UNKNOWN")
    {
        img_data_header->modality = MODALITY_UNDEFINED;
    }

    StrNumConverter<int> str_num_converter;
    img_data_header->patient_id = str_num_converter.to_string(meta_obj_tag->parent_id);

    img_data_header->sample_per_pixel = 1;
    img_data_header->pixel_spacing[0] = meta_obj_tag->spacing[0];
    img_data_header->pixel_spacing[1] = meta_obj_tag->spacing[1];
    
    img_data_header->rows = meta_obj_tag->dim_size[1];
    img_data_header->columns = meta_obj_tag->dim_size[0];

    //series
    int postfix_sub = -1;
    for (int i = int(meta_obj_tag->element_data_file.size())-1 ; i>= 0  ; --i)
    {
        if (postfix_sub == -1 && meta_obj_tag->element_data_file[i] == '.')
        {
            postfix_sub = i;
            break;
        }
    }
    if (postfix_sub != -1 && 
        (meta_obj_tag->element_data_file.size() - postfix_sub == 4 ||//"raw" 
        meta_obj_tag->element_data_file.size() - postfix_sub == 5) ) //"zraw"
    {
        img_data_header->series_uid = meta_obj_tag->element_data_file.substr(0 , postfix_sub);
    }

    //TODO calculate image position and slice location of each slice (necessarily ?)

    return IO_SUCCESS;
}

IOStatus MetaObjectLoader::construct_image_data_i( 
    const std::string& info_file ,
    std::shared_ptr<MetaObjectTag> meta_obj_tag , 
    std::shared_ptr<ImageDataHeader> img_data_header, 
    std::shared_ptr<ImageData> img_data )
{
    int file_dir_sub = -1;
    for (int i = int(info_file.size()) -1 ; i >= 0 ;--i)
    {
        if (info_file[i] == '\\' || info_file[i] == '/')
        {
            file_dir_sub = i;
            break;
        }
    }

    if(file_dir_sub == -1)
    {
        return IO_FILE_OPEN_FAILED;
    }

    std::string raw_file = "";
    const std::string file1 = info_file.substr(0 , file_dir_sub+1) + meta_obj_tag->element_data_file;
    const std::string file2 = meta_obj_tag->element_data_file;

    std::ifstream in(file1 , std::ios::out | std::ios::binary);
    if (in.is_open())
    {
        raw_file = file1;
    }
    else
    {
        in.open(file2 , std::ios::out | std::ios::binary);
        if (in.is_open())
        {
            raw_file = file2;
        }
    }

    if(raw_file.empty())
    {
        return IO_FILE_OPEN_FAILED;
    }

    if (meta_obj_tag->is_compressed_data)
    {
        in.close();

        const unsigned int c_size = meta_obj_tag->compressed_data_size;
        std::unique_ptr<char[]> c_data(new char[c_size]);
        in.read(c_data.get() , c_size);
        in.close();

        img_data->mem_allocate();
        const unsigned int img_size = img_data->get_data_size();
        IOStatus status = ZLibUtils::decompress(raw_file , (char*)img_data->get_pixel_pointer() , img_size);
        return status;
    }
    else 
    {
        img_data->mem_allocate();
        const unsigned int img_size = img_data->get_data_size();
        in.read((char*)img_data->get_pixel_pointer() , img_size);
        in.close();
    }

    return IO_SUCCESS;
}

MED_IMG_END_NAMESPACE