#include "mi_dicom_loader.h"
#include "mi_dicom_rt_struct_loader.h"
#include "mi_dicom_rt_struct.h"


using namespace MedImaging;

namespace
{

}

void IOUT_LoadRTStructureSet()
{
    std::shared_ptr<RTStruct> pRT;

    //std::string sFile = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RS.277458.dcm";

    //std::string sFile1 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RD.277458.Dose_PLAN.dcm";
    //std::string sFile2 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RI.277458.1_125.dcm";
    //std::string sFile3 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RI.277458.2_126.dcm";
    //std::string sFile4 = "E:/瓦里安/NPC_VARIAN_/NPC_VARIAN_/RP.277458.Plan2 #.dcm";


    std::string sFile = "E:/Data/RT/chenhongming/RS.0000001.dcm";

    DICOMRTLoader loader;
    loader.LoadRTStruct(sFile , pRT);
    pRT->WriteToFile("D:/temp/chenhongming_Contour.txt");
    //loader.LoadRTStruct(sFile1);
    //loader.LoadRTStruct(sFile2);
    //loader.LoadRTStruct(sFile3);
    //loader.LoadRTStruct(sFile4);


}

