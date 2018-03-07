#ifndef MED_IMG_APPCOMMON_MI_MODEL_PACS_CACHE_H
#define MED_IMG_APPCOMMON_MI_MODEL_PACS_CACHE_H

#include <vector>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"

#include "io/mi_protobuf.h"
#include "io/mi_dicom_info.h"

class MsgStudyWrapperCollection;

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelPACSCache : public IModel { 
public:
    ModelPACSCache();
    ~ModelPACSCache();

    void update(MsgStudyWrapperCollection& msg);

    void get_study_infos(int start, int end, 
        std::vector<MsgStudyInfo*>& study_infos, 
        std::vector<MsgPatientInfo*>& patient_infos);

    void get_series_info(int study_idx, std::vector<MsgSeriesInfo*>& series_infos);
    
    int get_study_series_uid(int study_idx, int series_idx, std::string& study_uid, std::string& series_uid);

    void print_all_series();

private:
    std::vector<MsgStudyInfo> _study_infos;
    std::vector<MsgPatientInfo> _patient_infos;
    std::vector<std::vector<MsgSeriesInfo>> _series_infos;

private:
    DISALLOW_COPY_AND_ASSIGN(ModelPACSCache);
};

MED_IMG_END_NAMESPACE
#endif