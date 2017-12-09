#include "mi_model_pacs.h"

#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

ModelPACS::ModelPACS() {

}

ModelPACS::~ModelPACS() {

}

void ModelPACS::clear() {
    _dcm_infos.clear();
    _series_mapper.clear();
}

void ModelPACS::insert(DcmInfo dcm_info) {
    auto it = _series_mapper.find(dcm_info.series_id);
    if (it == _series_mapper.end()) {
        MI_APPCOMMON_LOG(MI_WARNING) << "insert series already exist in PACS model, replace old one.";
        _dcm_infos[it->second] = dcm_info;
    } else {
        _dcm_infos.push_back(dcm_info);
        _series_mapper[dcm_info.series_id] = (int)_dcm_infos.size()-1;
    }
}

void ModelPACS::update(const std::vector<DcmInfo>& dcm_infos) {
    _dcm_infos = dcm_infos;
    _series_mapper.clear();
    int id=0;
    for(auto it=dcm_infos.begin(); it!=dcm_infos.end(); ++it) {
        const DcmInfo& dcm_info = *it;
        _series_mapper[dcm_info.series_id] = id++;
    }
}

int ModelPACS::query_dicom(int idx, DcmInfo& dcm_info) {
    if (idx < 0 || idx > (int)_dcm_infos.size()-1) {
        return -1;
    } else {
        dcm_info = _dcm_infos[idx];
        return 0;
    }
}

int ModelPACS::query_dicom(const std::string& series_id, DcmInfo& dcm_info) {
    auto it = _series_mapper.find(series_id);
    if (it == _series_mapper.end()) {
        return -1;
    } else {
        dcm_info = _dcm_infos[it->second];
        return 0;
    }
}

void ModelPACS::sort(SortStrategy strategy, std::vector<DcmInfo>& result) {
    //TODO
}

void ModelPACS::filter(FilterStrategy strategy, std::string attribute, std::vector<DcmInfo>& result) {
    //TODO
}

MED_IMG_END_NAMESPACE