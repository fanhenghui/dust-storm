#ifndef MEDIMGIO_IMAGE_DATA_HEADER_H
#define MEDIMGIO_IMAGE_DATA_HEADER_H

#include "io/mi_io_export.h"
#include <string>
#include <vector>

#include "arithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE

//////////////////////////////////////////////////////////////////////////
// The image data header contains basic tag info,including :
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
//(0020,000D) UI Study instance UID
//(0020,000E) UI Series instance UID
//(0020,0013) IS instance Number
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

// VolumeInfo contain extern information of volume
enum Modality {       // 0008,0060
    CR,                 // Computed Radiography
    CT,                 // Computed Tomography
    MR,                 // Magnetic Resonance
    PT,                 // Positron emission tomography(PET)
    RT_STRUCT,          // RT structure set
    MODALITY_UNDEFINED, // Other modality supported yet!
};

enum PatientPosition { // 0018,5100
    HFP,               //头在前 俯卧
    HFS,               //头在前 仰卧
    HFDR,              //头在前 右侧卧
    HFDL,              //头在前 左侧卧
    FFP,               //脚在前 俯卧
    FFS,               //脚在前 仰卧
    FFDR,              //脚在前 右侧卧
    FFDL               //脚在前 左侧S卧
};

enum PhotometricInterpretation { // 0028,0004 像素格式
    PI_MONOCHROME1, //单色，且最小值被映射为白色，且Sample per
    // pixel(0028,0002)值为1
    PI_MONOCHROME2, //单色，且最小值被映射为黑色，且Sample per
    // pixel(0028,0002)值为1
    PI_RGB,         // RGB三通道彩色，且Sample per pixel(0028,0002)值为3
    // MI_PI_PALETTE_COLOR,// 暂时不支持
    // MI_PI_YBR_FULL,// 暂时不支持
};

class ImageDataHeader {
public:
    //数据源
    // 0 DICOM
    // 1 Meta image
    unsigned int data_source_uid;

    //(0002,0010) UI Transfer Syntax UID
    std::string transfer_syntax_uid;

    //(0008,0023) TM Content Time(formerly known as image date)
    //序列生成时间（年月日如：20120818）
    std::string image_date;

    //(0008,0060) CS Modality(CT MRI PET)
    Modality modality;

    //(0008,0070) LO Manufacturer 厂商（如Philps）
    std::string manufacturer;

    //(0008,1090) LO Manufacturer's Model Name 厂商的型号名称（如 Brilliance 64）
    std::string manufacturer_model_name;

    //(0018,0060) DS KVP
    // KV 千伏 指电压 p 峰值(如120)
    float kvp;

    //(0010,0010) PN Patient's Name 病人名字（如ZhangSan）
    std::string patient_name;

    //(0010,0020) LO Patient's ID 病人ID（如AB_CTA_01）
    std::string patient_id;

    //(0010,0040) CS Patient's Sex 病人性别（如M）
    std::string patient_sex;

    //(0010,1010) AS Age of Patient 病人年龄（格式：nnnD nnnW nnnM nnnY 表示
    //天周月岁 ， 如018M表示18个月）
    std::string patient_age;

    //(0018,5100) CS Patient Position 病人体位（如HFS）
    PatientPosition patient_position;

    //(0018,0050) DS Slice thickness  层间距
    double slice_thickness;

    //(0028,0010) US Rows 扫描行数 对应dim[1]
    unsigned int rows;

    //(0028,0011) USColumns 扫描列数 对应dim[0]
    unsigned int columns;

    //(0028,0030) DS Pixel Spacing 一张图像的spacing（每个slice应该一样）
    double pixel_spacing[2]; // Rows and Columns spacing

    //(0020,000D) UI Study instance UID
    std::string study_uid;

    //(0020,000E) UI Series instance UID
    std::string series_uid;

    //(0020,0032) DS Image Position(Patient)
    //图像位置，图像左上角的第一个像素的空间坐标(每个slice都不同)
    std::vector<Point3> image_position;

    //(0020,1041) DS Slice Location 图像切面位置(每个slice都不同)
    std::vector<double> slice_location;

    //(0028,0002) US Samples per Pixel
    unsigned int sample_per_pixel;

    //(0028,0004) CS Photometric Interpretation  数据类型解释（单值
    //三通道值还是带有颜色表）
    PhotometricInterpretation photometric_interpretation;

    //(0028,0100) US Bits Allocated 数据的存储位数
    unsigned int bits_allocated;

    //(0028,0103) US Pixel Representation 数据符号位
    // 0 unsigned
    // 1 2s complement (signed)
    unsigned int pixel_representation;
    
    bool reverse_z; // when loading the slices, whether z is reversed or not

    ImageDataHeader() {
        data_source_uid = 0;
        const std::string sUD = std::string("Undefined");
        image_date = sUD;
        modality = MODALITY_UNDEFINED;
        manufacturer = sUD;
        manufacturer_model_name = sUD;
        kvp = 0.0f;
        patient_name = sUD;
        patient_id = sUD;
        patient_sex = std::string("");
        patient_age = std::string("000D");
        patient_position = HFP;
        slice_thickness = 1.0;
        study_uid = sUD;
        series_uid = sUD;
        sample_per_pixel = 0;
        photometric_interpretation = PI_MONOCHROME2;
        bits_allocated = 0;
        pixel_representation = 0;
        reverse_z = false;
    }
};

MED_IMG_END_NAMESPACE

#endif