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
    unsigned int m_uiDataSrouceUID;

    //(0002,0010) UI Transfer Syntax UID
    std::string m_sTransferSyntaxUID;

    //(0008,0023) TM Content Time(formerly known as image date)
    //��������ʱ�䣨�������磺20120818��
    std::string m_sImageDate;

    //(0008,0060) CS Modality(CT MRI PET)
    Modality m_eModality;

    //(0008,0070) LO Manufacturer ���̣���Philps��
    std::string m_sManufacturer;

    //(0008,1090) LO Manufacturer's Model Name ���̵��ͺ����ƣ��� Brilliance 64��
    std::string m_sManufacturerModelName;

    //(0018,0060) DS KVP
    //KV ǧ�� ָ��ѹ p ��ֵ(��120)
    float m_kVp; 

    //(0010,0010) PN Patient's Name �������֣���ZhangSan��
    std::string m_sPatientName;

    //(0010,0020) LO Patient's ID ����ID����AB_CTA_01��
    std::string m_sPatientID;

    //(0010,0040) CS Patient's Sex �����Ա���M��
    std::string m_sPatientSex;

    //(0010,1010) AS Age of Patient �������䣨��ʽ��nnnD nnnW nnnM nnnY ��ʾ �������� �� ��018M��ʾ18���£�
    std::string m_sPatientAge;

    //(0018,5100) CS Patient Position ������λ����HFS��
    PatientPosition m_ePatientPosition;

    //(0018,0050) DS Slice thickness  ����
    double m_dSliceThickness;

    //(0028,0010) US Rows ɨ������ ��Ӧdim[1]
    unsigned int m_uiImgRows;

    //(0028,0011) USColumns ɨ������ ��Ӧdim[0]
    unsigned int m_uiImgColumns;

    //(0028,0030) DS Pixel Spacing һ��ͼ���spacing��ÿ��sliceӦ��һ����
    double m_dPixelSpacing[2];//Rows and Columns spacing

    //(0020,000D) UI Study Instance UID
    std::string m_sStudyUID;

    //(0020,000E) UI Series Instance UID
    std::string m_sSeriesUID;

    //(0020,0032) DS Image Position(Patient) ͼ��λ�ã�ͼ�����Ͻǵĵ�һ�����صĿռ�����(ÿ��slice����ͬ)
    std::vector<Point3> m_ImgPositions;

    //(0020,1041) DS Slice Location ͼ������λ��(ÿ��slice����ͬ)
    std::vector<double>  m_SliceLocations;

    //(0028,0002) US Samples per Pixel
    unsigned int m_uiSamplePerPixel;

    //(0028,0004) CS Photometric Interpretation  �������ͽ��ͣ���ֵ ��ͨ��ֵ���Ǵ�����ɫ��
    PhotometricInterpretation m_ePhotometricInterpretation;

    //(0028,0100) US Bits Allocated ���ݵĴ洢λ��
    unsigned int m_uiBitsAllocated;

    //(0028,0103) US Pixel Representation ���ݷ���λ
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