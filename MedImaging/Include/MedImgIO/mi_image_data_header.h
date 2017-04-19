#ifndef MED_IMAGING_IMAGE_DATA_HEADER_H
#define MED_IMAGING_IMAGE_DATA_HEADER_H

#include "MedImgIO/mi_io_stdafx.h"
#include <string>
#include <vector>

#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

//////////////////////////////////////////////////////////////////////////
//The image data header contains basic tag info,including :
//(0002,0010) UI Transfer Syntax UID  
//(0008,0023) TM Content Time(formerly known as image date)
//(0008,0060) CS Modality
//(0008,0070) LO Manufacturer
//(0008,1090) LO Manufacturer's Model Name
//(0010,0010) PN Patient's Name
//(0010,0020) LO Patient's ID
//(0010,0040) CS Patient's Sex
//(0010,1010) AS Age of Patient
//(0018,0050) DS Slice thickness 
//(0018,0060) DS KVP
//(0018,5100) CS Patient Position 
//(0020,000D) UI Study Instance UID
//(0020,000E) UI Series Instance UID
//(0020,0013) IS Instance Number
//(0020,0032) DS Image Position(Patient)
//(0020,0037) DS Image Orientation(Patient)
//(0020,1041) DS Slice Location
//(0028,0002) US Samples per Pixel
//(0028,0004) CS Photometric Interpretation 
//(0028,0010) US Rows
//(0028,0011) US Columns
//(0028,0030) DS Pixel Spacing
//(0028,0100) US Bits Allocated
//(0028,0103) US Pixel Representation
//(0028,1052) DS Rescale Intercept
//(0028,1053) DS Rescale Slope
//(7FE0,0010) OW/OB Pixel data
//////////////////////////////////////////////////////////////////////////

//VolumeInfo contain extern information of volume
enum Modality//0008,0060
{
    CR,//Computed Radiography
    CT,//Computed Tomography
    MR,//Magnetic Resonance
    PT,//Positron emission tomography(PET)
    RT_STRUCT,//RT structure set
    MODALITY_UNDEFINED,//Other modality supported yet!
};

enum PatientPosition//0018,5100
{
    HFP,//头在前 俯卧
    HFS,//头在前 仰卧
    HFDR,//头在前 右侧卧
    HFDL,//头在前 左侧卧
    FFP,//脚在前 俯卧
    FFS,//脚在前 仰卧
    FFDR,//脚在前 右侧卧
    FFDL//脚在前 左侧S卧
};

enum PhotometricInterpretation //0028,0004 像素格式
{
    PI_MONOCHROME1,//单色，且最小值被映射为白色，且Sample per pixel(0028,0002)值为1
    PI_MONOCHROME2,//单色，且最小值被映射为黑色，且Sample per pixel(0028,0002)值为1
    PI_RGB,//RGB三通道彩色，且Sample per pixel(0028,0002)值为3
    //MI_PI_PALETTE_COLOR,// 暂时不支持
    //MI_PI_YBR_FULL,// 暂时不支持
};

class ImageDataHeader
{
public:
    //数据源
    //0 DICOM
    //1 Meta image
    unsigned int m_uiDataSrouceUID;

    //(0002,0010) UI Transfer Syntax UID
    std::string m_sTransferSyntaxUID;

    //(0008,0023) TM Content Time(formerly known as image date)
    //序列生成时间（年月日如：20120818）
    std::string m_sImageDate;

    //(0008,0060) CS Modality(CT MRI PET)
    Modality m_eModality;

    //(0008,0070) LO Manufacturer 厂商（如Philps）
    std::string m_sManufacturer;

    //(0008,1090) LO Manufacturer's Model Name 厂商的型号名称（如 Brilliance 64）
    std::string m_sManufacturerModelName;

    //(0018,0060) DS KVP
    //KV 千伏 指电压 p 峰值(如120)
    float m_kVp; 

    //(0010,0010) PN Patient's Name 病人名字（如ZhangSan）
    std::string m_sPatientName;

    //(0010,0020) LO Patient's ID 病人ID（如AB_CTA_01）
    std::string m_sPatientID;

    //(0010,0040) CS Patient's Sex 病人性别（如M）
    std::string m_sPatientSex;

    //(0010,1010) AS Age of Patient 病人年龄（格式：nnnD nnnW nnnM nnnY 表示 天周月岁 ， 如018M表示18个月）
    std::string m_sPatientAge;

    //(0018,5100) CS Patient Position 病人体位（如HFS）
    PatientPosition m_ePatientPosition;

    //(0018,0050) DS Slice thickness  层间距
    double m_dSliceThickness;

    //(0028,0010) US Rows 扫描行数 对应dim[1]
    unsigned int m_uiImgRows;

    //(0028,0011) USColumns 扫描列数 对应dim[0]
    unsigned int m_uiImgColumns;

    //(0028,0030) DS Pixel Spacing 一张图像的spacing（每个slice应该一样）
    double m_dPixelSpacing[2];//Rows and Columns spacing

    //(0020,000D) UI Study Instance UID
    std::string m_sStudyUID;

    //(0020,000E) UI Series Instance UID
    std::string m_sSeriesUID;

    //(0020,0032) DS Image Position(Patient) 图像位置，图像左上角的第一个像素的空间坐标(每个slice都不同)
    std::vector<Point3> m_ImgPositions;

    //(0020,1041) DS Slice Location 图像切面位置(每个slice都不同)
    std::vector<double>  m_SliceLocations;

    //(0028,0002) US Samples per Pixel
    unsigned int m_uiSamplePerPixel;

    //(0028,0004) CS Photometric Interpretation  数据类型解释（单值 三通道值还是带有颜色表）
    PhotometricInterpretation m_ePhotometricInterpretation;

    //(0028,0100) US Bits Allocated 数据的存储位数
    unsigned int m_uiBitsAllocated;

    //(0028,0103) US Pixel Representation 数据符号位
    // 0 unsigned
    // 1 2s complement (signed)
    unsigned int m_uiPixelRepresentation;

    ImageDataHeader()
    {
        m_uiDataSrouceUID = 0;
        const std::string sUD = std::string("Undefined");
        m_sImageDate = sUD;
        m_eModality = MODALITY_UNDEFINED;
        m_sManufacturer = sUD;
        m_sManufacturerModelName = sUD;
        m_kVp = 0.0f;
        m_sPatientName = sUD;
        m_sPatientID = sUD;
        m_sPatientSex = std::string("M");
        m_sPatientAge = std::string("000D");
        m_ePatientPosition = HFP;
        m_dSliceThickness = 1.0;
        m_sStudyUID = sUD;
        m_sSeriesUID = sUD;
        m_uiSamplePerPixel = 0;
        m_ePhotometricInterpretation = PI_MONOCHROME2;
        m_uiBitsAllocated = 0;
        m_uiPixelRepresentation = 0;
    }

};

MED_IMAGING_END_NAMESPACE

#endif