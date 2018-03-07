#include "mi_model_pacs_cache.h"
#include "io/mi_protobuf.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

ModelPACSCache::ModelPACSCache() {

}

ModelPACSCache::~ModelPACSCache() {
    
}

void ModelPACSCache::update(MsgStudyWrapperCollection& msg) {
    _study_infos.clear();
    _patient_infos.clear();
    _series_infos.clear();

    const int study_size = msg.study_wrappers_size() <= 0 ? 0 : msg.study_wrappers_size();
    for (int i = 0; i < study_size; ++i) {
        const MsgStudyWrapper& study_wrapper = msg.study_wrappers(i);
        const MsgStudyInfo& study_info = study_wrapper.study_info();
        const MsgPatientInfo& patient_info = study_wrapper.patient_info();

        _study_infos.push_back(study_info);
        _study_infos[_study_infos.size()-1].set_id(i);
        _patient_infos.push_back(patient_info);
        _series_infos.push_back(std::vector<MsgSeriesInfo>());
        int series_size = study_wrapper.series_infos_size();
        for (int j = 0; j < series_size; ++j) {
            const MsgSeriesInfo& series_info = study_wrapper.series_infos(j);
            _series_infos[i].push_back(series_info);
            _series_infos[i][j].set_id(j);
        }
    }
}

void ModelPACSCache::get_study_infos(int start, int end, 
    std::vector<MsgStudyInfo*>& study_infos, 
    std::vector<MsgPatientInfo*>& patient_infos) {
    study_infos.clear();
    patient_infos.clear();
    for (int i = start; i < end ; ++i) {
        if (i > (int)_study_infos.size() -1) {
            break;
        }
        study_infos.push_back(&(_study_infos[i]));
        patient_infos.push_back(&(_patient_infos[i]));
    }
}

void ModelPACSCache::get_series_info(int study_idx, std::vector<MsgSeriesInfo*>& series_infos) {
    if (study_idx < (int)_series_infos.size()) {
        series_infos.clear();
        for (size_t i = 0; i < _series_infos[study_idx].size(); ++i) {
            series_infos.push_back(&(_series_infos[study_idx][i]));
        }
    }
}

int ModelPACSCache::get_study_series_uid(int study_idx, int series_idx, std::string& study_uid, std::string& series_uid) {
    if (study_idx > (int)_series_infos.size()-1) { 
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid study idx: " << study_idx;
        return -1;
    }
    const std::vector<MsgSeriesInfo>& series_infos = _series_infos[study_idx];
    if (series_idx > (int)series_infos.size() - 1) {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid series idx: " << series_idx;
        return -1;
    }
    
    study_uid = _study_infos[study_idx].study_uid();
    series_uid = series_infos[series_idx].series_uid();
    return 0;
}

void ModelPACSCache::print_all_series() {
    for (size_t i = 0; i < _study_infos.size(); ++i) {
        const MsgStudyInfo& study_info = _study_infos[i];
        const MsgPatientInfo& patient_info = _patient_infos[i];

        for (size_t j = 0; j < _series_infos[i].size(); ++j) {
            const MsgSeriesInfo& series_info = _series_infos[i][j];
            MI_APPCOMMON_LOG(MI_DEBUG) << i <<": " << std::endl
            << "study_uid: " << study_info.study_uid() << std::endl
            << "study_id: " << study_info.study_id() << std::endl
            << "study_date: " << study_info.study_date() << std::endl
            << "study_time: " << study_info.study_time() << std::endl
            << "accession_no: " << study_info.accession_no() << std::endl
            << "study_desc: " << study_info.study_desc() << std::endl
            << "num_instance(study): " << study_info.num_instance() << std::endl
            << "num_series: " << study_info.num_series() << std::endl
            << "series_uid: " << series_info.series_uid() << std::endl
            << "series_no: " << series_info.series_no() << std::endl
            << "modality: " << series_info.modality() << std::endl
            << "series_desc: " << series_info.series_desc() << std::endl
            << "institution: " << series_info.institution() << std::endl
            << "num_instance(series): " << series_info.num_instance() << std::endl
            << "patient_id: " << patient_info.patient_id() << std::endl
            << "patient_name: " << patient_info.patient_name() << std::endl
            << "patient_sex: " << patient_info.patient_sex() << std::endl
            << "patient_birth_date: " << patient_info.patient_birth_date() << std::endl
            << std::endl;
        }
    }
}

MED_IMG_END_NAMESPACE
