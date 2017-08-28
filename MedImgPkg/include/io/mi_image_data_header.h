#ifndef MED_IMG_IMAGE_DATA_HEADER_H
#define MED_IMG_IMAGE_DATA_HEADER_H

#include "io/mi_io_export.h"
#include <string>
#include <vector>

#include "arithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE 

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
    HFP,//ͷ��ǰ ����
    HFS,//ͷ��ǰ ����
    HFDR,//ͷ��ǰ �Ҳ���
    HFDL,//ͷ��ǰ �����
    FFP,//����ǰ ����
    FFS,//����ǰ ����
    FFDR,//����ǰ �Ҳ���
    FFDL//����ǰ ���S��
};

enum PhotometricInterpretation //0028,0004 ���ظ�ʽ
{
    PI_MONOCHROME1,//��ɫ������Сֵ��ӳ��Ϊ��ɫ����Sample per pixel(0028,0002)ֵΪ1
    PI_MONOCHROME2,//��ɫ������Сֵ��ӳ��Ϊ��ɫ����Sample per pixel(0028,0002)ֵΪ1
    PI_RGB,//RGB��ͨ����ɫ����Sample per pixel(0028,0002)ֵΪ3
    //MI_PI_PALETTE_COLOR,// ��ʱ��֧��
    //MI_PI_YBR_FULL,// ��ʱ��֧��
};

class ImageDataHeader
{
public:
    //����Դ
    //0 DICOM
    //1 Meta image
    unsigned int data_source_uid;

    //(0002,0010) UI Transfer Syntax UID
    std::string transfer_syntax_uid;

    //(0008,0023) TM Content Time(formerly known as image date)
    //��������ʱ�䣨�������磺20120818��
    std::string image_date;

    //(0008,0060) CS Modality(CT MRI PET)
    Modality modality;

    //(0008,0070) LO Manufacturer ���̣���Philps��
    std::string manufacturer;

    //(0008,1090) LO Manufacturer's Model Name ���̵��ͺ����ƣ��� Brilliance 64��
    std::string manufacturer_model_name;

    //(0018,0060) DS KVP
    //KV ǧ�� ָ��ѹ p ��ֵ(��120)
    float kvp; 

    //(0010,0010) PN Patient's Name �������֣���ZhangSan��
    std::string patient_name;

    //(0010,0020) LO Patient's ID ����ID����AB_CTA_01��
    std::string patient_id;

    //(0010,0040) CS Patient's Sex �����Ա���M��
    std::string patient_sex;

    //(0010,1010) AS Age of Patient �������䣨��ʽ��nnnD nnnW nnnM nnnY ��ʾ �������� �� ��018M��ʾ18���£�
    std::string patient_age;

    //(0018,5100) CS Patient Position ������λ����HFS��
    PatientPosition patient_position;

    //(0018,0050) DS Slice thickness  ����
    double slice_thickness;

    //(0028,0010) US Rows ɨ������ ��Ӧdim[1]
    unsigned int rows;

    //(0028,0011) USColumns ɨ������ ��Ӧdim[0]
    unsigned int columns;

    //(0028,0030) DS Pixel Spacing һ��ͼ���spacing��ÿ��sliceӦ��һ����
    double pixel_spacing[2];//Rows and Columns spacing

    //(0020,000D) UI Study instance UID
    std::string study_uid;

    //(0020,000E) UI Series instance UID
    std::string series_uid;

    //(0020,0032) DS Image Position(Patient) ͼ��λ�ã�ͼ�����Ͻǵĵ�һ�����صĿռ�����(ÿ��slice����ͬ)
    std::vector<Point3> image_position;

    //(0020,1041) DS Slice Location ͼ������λ��(ÿ��slice����ͬ)
    std::vector<double>  slice_location;

    //(0028,0002) US Samples per Pixel
    unsigned int sample_per_pixel;

    //(0028,0004) CS Photometric Interpretation  �������ͽ��ͣ���ֵ ��ͨ��ֵ���Ǵ�����ɫ��
    PhotometricInterpretation photometric_interpretation;

    //(0028,0100) US Bits Allocated ���ݵĴ洢λ��
    unsigned int bits_allocated;

    //(0028,0103) US Pixel Representation ���ݷ���λ
    // 0 unsigned
    // 1 2s complement (signed)
    unsigned int pixel_representation;

    ImageDataHeader()
    {
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
    }

};

MED_IMG_END_NAMESPACE

#endif