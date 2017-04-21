#include "mi_dicom_loader.h"
#include "mi_dicom_rt_struct_loader.h"
#include "mi_dicom_rt_struct.h"


using namespace medical_imaging;

namespace
{

}

void IOUT_LoadRTStructureSet()
{
    std::shared_ptr<RTStruct> pRT;

    //std::string file_name = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RS.277458.dcm";

    //std::string sFile1 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RD.277458.Dose_PLAN.dcm";
    //std::string sFile2 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RI.277458.1_125.dcm";
    //std::string sFile3 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RI.277458.2_126.dcm";
    //std::string sFile4 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RP.277458.Plan2 #.dcm";


    std::string file_name = "E:/Data/RT/chenhongming/RS.0000001.dcm";

    DICOMRTLoader loader;
    loader.load_rt_struct(file_name , pRT);
    pRT->write_to_file("D:/temp/chenhongming_Contour.txt");
    //loader.load_rt_struct(sFile1);
    //loader.load_rt_struct(sFile2);
    //loader.load_rt_struct(sFile3);
    //loader.load_rt_struct(sFile4);


}

