#ifndef MED_IMG_META_OBJECTLOADER_H
#define MED_IMG_META_OBJECTLOADER_H

#include "io/mi_io_export.h"
#include "io/mi_io_define.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_vector3.h"

MED_IMG_BEGIN_NAMESPACE 

class ImageData;
class ImageDataHeader;

#define META_COMMENT "Comment"
#define META_OBJECT_TYPE "ObjectType"
#define META_OBJECT_SUB_TYPE "ObjectSubType"
#define META_TRANSFORM_TYPE "TransformType"
#define META_NDIMS "NDims"
#define META_NAME "Name"
#define META_ID "ID"
#define META_PATIENT_ID "ParentID"
#define META_BINARY_DATA "BinaryData"
#define META_ELEMENT_BYTE_ORDER_MSB "ElementByteOrderMSB"
#define META_BINARY_DATA_BYTE_ORDER_MSB "BinaryDataByteOrderMSB"
#define META_COMPRESSD_DATA "CompressedData"
#define META_COMPRESSED_DATA_SIZE "CompressedDataSize "
#define META_COLOR "Color"

//Position: (equiv. to offset and origin)
#define META_POSITION "Position"
#define META_OFFSET "Offset"
#define META_ORIGIN "Origin"

//Rotation: (equiv. to orientation and transformMatrix)
#define META_ROTATION "Rotation"
#define META_ORIENTATION "Orientation"
#define META_TRANSFORM_MATRIX "TransformMatrix"

#define META_ANATOMICAL_ORIENTATION "AnatomicalOrientation"
#define META_ELEMENT_SPACING "ElementSpacing"
#define META_DIM_SIZE "DimSize"
#define META_HEADER_SIZE "HeaderSize"
#define META_MODALITY "Modality"
#define META_SEQUENCE_ID "SequenceID"
#define META_ELEMENT_MIN "ElementMin"
#define META_ELEMENT_MAX "ElementMax"
#define META_ELEMENT_NUMBER_OF_CHANNELS "ElementNumberOfChannels"
#define META_ELEMENT_SIZE "ElementSize"
#define META_ELEMENT_TYPE "ElementType"
#define META_ELEMENT_DATA_FILE "ElementDataFile"


//Meta Tag info from url :
//https://itk.org/Wiki/MetaIO/Documentation#MetaObject_Tags
struct MetaObjectTag
{
    //////////////////////////////////////////////////////////////////////////
    //MetaObject Tags
    //////////////////////////////////////////////////////////////////////////

    //Comment MET_STRING User defined - arbitrary
    std::string comment;

    //ObjectType MET_STRING Defined by derived objects �C e.g., Tube, Image
    std::string object_type;

    //ObjectSubType MET_STRING Defined by derived objects �C currently not used
    std::string object_sub_type;

    //TransformType MET_STRING Defined by derived objects �C e.g., Rigid
    std::string transform_type;

    //NDims MET_INT Defined at object instantiation
    int n_dims;//�������ֻ֧�ֵ�3

    //Name MET_STRING User defined
    std::string name;

    //ID MET_INT User defined else -1
    int id;

    //ParentID MET_INT User defined else -1
    int parent_id;

    //BinaryData MET_STRING Are the data associated with this object stored at Binary or ASCII
    //Defined by derived objects- e.g.,  True
    bool is_binary_data;

    //ElementByteOrderMSB  MET_STRING
    std::string element_byte_order_msb;

    //BinaryDataByteOrderMSB MET_STRING
    std::string binary_data_byte_order_msb;

    //CompressedData True or False
    bool is_compressed_data;

    //CompressedDataSize  MET_FLOAT
    unsigned int compressed_data_size;

    //Color MET_FLOAT_ARRAY[4] R, G, B, alpha (opacity)
    float color[4];

    //Position/Offset/Origin MET_FLOAT_ARRAY[NDims]
    //X, Y, Z,�� of real-world coordinate of 0,0,0 index of image)
    Point3 position;

    //Orientation/Rotation/TransformMatrix MET_FLOAT_MATRIX[NDims][NDims]
    Vector3 orientation_x;
    Vector3 orientation_y;
    Vector3 orientation_z;

    //AnatomicalOrientation MET_STRING
    //Specify anatomic ordering of the axis. Use only [R|L] | [A|P] | [S|I] per axis. 
    //For example : if the three letter code for (column index, row index, slice index is) ILP, 
    //                          then the origin is at the superior, right, anterior corner of the volume, 
    //                          and therefore the axes run from superior to inferior, from right to left, from anterior to posterior.
    std::string anatomical_orientation;

    //ElementSpacing MET_FLOAT_ARRAY[NDims] The distance between voxel centers
    double spacing[3];

    //////////////////////////////////////////////////////////////////////////
    //Tags Added by MetaImage
    //////////////////////////////////////////////////////////////////////////

    //DimSize MET_INT_ARRAY[NDims] Number of elements per axis in data
    unsigned int dim_size[3];

    //HeaderSize MET_INT
    //Number of Bytes to skip at the head of each data file.
    //Specify �C1 to have MetaImage calculate the header size based on the assumption that the data occurs at the end of the file.
    //Specify 0 if the data occurs at the begin of the file.
    int header_size;

    //Modality MET_STRING
    //One of enum type: MET_MOD_CT, MET_MOD_MR, MET_MOD_US , MET_MOD_OTHER , MET_MOD_UNKNOWN
    std::string modality;

    //SequenceID MET_INT_ARRAY[4]
    //Four values comprising a DICOM sequence: Study, Series, Image numbers
    int sequence_id[4];

    //ElementMin MET_FLOAT
    //Minimum value in the data
    float element_min;

    //ElementMax MET_FLOAT
    //Maximum value in the data
    float element_max;

    //ElementNumberOfChannels MET_INT
    //Number of values (of type ElementType) per voxel
    int element_number_of_channels;

    //ElementSize
    //MET_FLOAT_ARRAY[NDims]
    //Physical size of each voxel
    double element_size[3];

    //ElementType MET_STRING
    //One of enum type: MET_UCHAR, MET_CHAR , MET_USHORT ,MET_SHORT ,MET_INT,MET_UINT ,MET_FLOAT...
    std::string element_type;

    //ElementDataFile  MET_STRING
    //One of the following:
    //1 Name of the file to be loaded
    //2 A printf-style string followed by the min, max, and step values to be used to pass an argument to the string to create list of file names to be loaded (must be (N-1)D blocks of data per file).
    //3 LIST [X] �C This specifies that starting on the next line is a list of files (one filename per line) in which the data is stored. Each file (by default) contains an (N-1)D block of data. If a second argument is given, its first character must be a number that specifies the dimension of the data in each file. For example ElementDataFile = LIST 2D means that there will be a 2D block of data per file.
    //4 LOCAL �C Indicates that the data begins at the beginning of the next line.
    std::string element_data_file;

    MetaObjectTag()
    {
        n_dims = 1;
        id = 0;
        parent_id = 0;
        is_compressed_data = false;
        compressed_data_size = 0;
        is_binary_data = false;
        color[0] = color[1] = color[2] = color[3] = 0; 
        spacing[0] = spacing[1] = spacing[2] = 0;
        dim_size[0] = dim_size[1] = dim_size[2] = 0;
        header_size = 0;
        modality = "MET_MOD_UNKNOWN";
        sequence_id[0] = sequence_id[1] = sequence_id[2] = sequence_id[3] = 0;
        element_min = element_max = 0;
        element_number_of_channels = 1;
        element_size[0] = element_size[1] = element_size[2] = 0;
        element_type = "MET_UCHAR";
        element_data_file = "";
    }
};

class IO_Export MetaObjectLoader
{
public:
    IOStatus load(
        const std::string& info_file ,
        std::shared_ptr<ImageData> &image_data , 
        std::shared_ptr<MetaObjectTag> & meta_obj_tag,
        std::shared_ptr<ImageDataHeader> & img_data_header);

private:
    IOStatus construct_meta_object_tag_i(
        const std::string& info_file , 
        std::shared_ptr<MetaObjectTag> & meta_obj_tag);

    IOStatus construct_data_header_i(
        std::shared_ptr<MetaObjectTag> meta_obj_tag , 
        std::shared_ptr<ImageDataHeader> & img_data_header,
        std::shared_ptr<ImageData> & img_data);

    IOStatus construct_image_data_i(
        const std::string& info_file ,
        std::shared_ptr<MetaObjectTag> meta_obj_tag , 
        std::shared_ptr<ImageDataHeader>  img_data_header,
        std::shared_ptr<ImageData> img_data);

};

MED_IMG_END_NAMESPACE
#endif