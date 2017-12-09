#ifndef MED_IMG_APPCOMMON_MI_MODEL_PACS_H
#define MED_IMG_APPCOMMON_MI_MODEL_PACS_H

#include <string>
#include <vector>
#include "io/mi_dicom_info.h"
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelPACS : public IModel { 
public:
    enum SortStrategy {
        SORT_STUDY_DATE,
        SORT_BIRTHDAY,
        SORT_PATIENT_ID,
    };

    enum FilterStrategy {
        FILTER_STUDY_DATE,
        FILTER_BIRTHDAY,
        FILTER_MODALITY,
    };

    ModelPACS();
    virtual ~ModelPACS();

    void clear();
    void insert(DcmInfo dcm_info);
    void update(const std::vector<DcmInfo>& dcm_infos);

    int  query_dicom(int idx, DcmInfo& dcm_info);    
    int  query_dicom(const std::string& series_id, DcmInfo& dcm_info);
    void sort(SortStrategy strategy, std::vector<DcmInfo>& result);
    void filter(FilterStrategy strategy, std::string attribute, std::vector<DcmInfo>& result);

private:
    std::vector<DcmInfo> _dcm_infos;
    std::map<std::string,int> _series_mapper;

    ////TODO cache for paging
    // int _sort_cache_id;
    // std::vector<DcmInfo> _sort_cache;
private:
    DISALLOW_COPY_AND_ASSIGN(ModelPACS);
};

MED_IMG_END_NAMESPACE
#endif