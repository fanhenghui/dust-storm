#ifndef MED_IMAGING_META_OBJECTLOADER_H
#define MED_IMAGING_META_OBJECTLOADER_H

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3.h"

MED_IMAGING_BEGIN_NAMESPACE

class ImageData;
class ImageDataHeader;

//Meta Tag info from web : 
//https://itk.org/Wiki/MetaIO/Documentation#MetaObject_Tags

struct MetaObjectTag
{
    //////////////////////////////////////////////////////////////////////////
    //MetaObject Tags
    //////////////////////////////////////////////////////////////////////////

    //Comment MET_STRING User defined - arbitrary
    std::string m_sComment;

    //ObjectType MET_STRING Defined by derived objects – e.g., Tube, Image
    std::string m_sObjectType;

    //ObjectSubType MET_STRING Defined by derived objects – currently not used
    std::string m_sObjectSubType;

    //TransformType MET_STRING Defined by derived objects – e.g., Rigid
    std::string m_sTransformType;

    //NDims MET_INT Defined at object instantiation
    int m_iNDims;//这里最大只支持到3

    //Name MET_STRING User defined
    std::string m_sName;

    //ID MET_INT User defined else -1
    int m_iID;

    //ParentID MET_INT User defined else -1
    int m_iParentID;

    //BinaryData MET_STRING Are the data associated with this object stored at Binary or ASCII
    //Defined by derived objects- e.g.,  True
    std::string m_sBinaryData;

    //ElementByteOrderMSB  MET_STRING
    std::string m_sElementByteOrderMSB;

    //BinaryDataByteOrderMSB MET_STRING
    std::string m_sBinaryDataByteOrderMSB;

    //Color MET_FLOAT_ARRAY[4] R, G, B, alpha (opacity)
    float m_fColor[4];

    //Position MET_FLOAT_ARRAY[NDims]
    //X, Y, Z,… of real-world coordinate of 0,0,0 index of image)
    Point3 m_ptPosition;

    //Orientation MET_FLOAT_MATRIX[NDims][NDims]
    Vector3 m_vOrientationX;
    Vector3 m_vOrientationY;

    //AnatomicalOrientation MET_STRING
    //Specify anatomic ordering of the axis. Use only [R|L] | [A|P] | [S|I] per axis. 
    //For example : if the three letter code for (column index, row index, slice index is) ILP, 
    //                          then the origin is at the superior, right, anterior corner of the volume, 
    //                          and therefore the axes run from superior to inferior, from right to left, from anterior to posterior.
    std::string m_sAnatomicalOrientation;

    //ElementSpacing MET_FLOAT_ARRAY[NDims] The distance between voxel centers
    double m_dSpacing[3];

    //////////////////////////////////////////////////////////////////////////
    //Tags Added by MetaImage
    //////////////////////////////////////////////////////////////////////////

    //DimSize MET_INT_ARRAY[NDims] Number of elements per axis in data
    unsigned int m_uiDimSize[3];

    //HeaderSize MET_INT
    //Number of Bytes to skip at the head of each data file.
    //Specify –1 to have MetaImage calculate the header size based on the assumption that the data occurs at the end of the file.
    //Specify 0 if the data occurs at the begining of the file.
    int m_iHeaderSize;

    //Modality MET_STRING
    //One of enum type: MET_MOD_CT, MET_MOD_MR, MET_MOD_US , MET_MOD_OTHER , MET_MOD_UNKNOWN
    std::string m_sModality;

    //SequenceID MET_INT_ARRAY[4]
    //Four values comprising a DICOM sequence: Study, Series, Image numbers
    int m_SequenceID[4];

    //ElementMin MET_FLOAT
    //Minimum value in the data
    float m_fElementMin;

    //ElementMax MET_FLOAT
    //Maximum value in the data
    float m_fElementMax;

    //ElementNumberOfChannels MET_INT
    //Number of values (of type ElementType) per voxel
    int m_iElementNumberOfChannels;

    //ElementSize
    //MET_FLOAT_ARRAY[NDims]
    //Physical size of each voxel
    double m_dElementSize[3];

    //ElementType MET_STRING
    //One of enum type: MET_UCHAR, MET_CHAR , MET_USHORT ,MET_SHORT ,MET_INT,MET_UINT ,MET_FLOAT...
    std::string m_sElementType;

    //ElementDataFile  MET_STRING
    //One of the following:
    //1 Name of the file to be loaded
    //2 A printf-style string followed by the min, max, and step values to be used to pass an argument to the string to create list of file names to be loaded (must be (N-1)D blocks of data per file).
    //3 LIST [X] – This specifies that starting on the next line is a list of files (one filename per line) in which the data is stored. Each file (by default) contains an (N-1)D block of data. If a second argument is given, its first character must be a number that specifies the dimension of the data in each file. For example ElementDataFile = LIST 2D means that there will be a 2D block of data per file.
    //4 LOCAL – Indicates that the data begins at the beginning of the next line.
    std::string m_sElementDataFile;
};

class IO_Export MetaObjectLoader
{
public:
    IOStatus load(
        const std::string& sInfoFile ,
        std::shared_ptr<ImageData> &pImgData , 
        std::shared_ptr<MetaObjectTag> & pMetaObjTag,
        std::shared_ptr<ImageDataHeader> & pImgDataHeader);

protected:
private:
};

MED_IMAGING_END_NAMESPACE
#endif