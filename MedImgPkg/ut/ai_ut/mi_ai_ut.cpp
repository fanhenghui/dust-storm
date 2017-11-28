
#include <string>
#include <time.h>

#include "log/mi_logger.h"
#include "util/mi_file_util.h"
#include "io/mi_nodule_set.h"
#include "io/mi_nodule_set_parser.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_app_db.h"
#include "mi_ai_lung_evaluate_py_wrapper.h" 

using namespace medical_imaging;

int main(int argc, char* argv[]) {

    if(argc != 2) {
        MI_LOG(MI_ERROR) << "invalid input."; 
        return -1;
    }
    const std::string series_id = argv[1];
    MI_LOG(MI_INFO) << "series id: " << series_id << std::endl;

    std::string ip_port,user,pwd,db_name;
    AppConfig::instance()->get_db_info(ip_port,user,pwd,db_name);
    DB db;
    if (-1 == db.connect(user,ip_port,pwd,db_name) ) {
        MI_LOG(MI_ERROR) << "connect DB failed.\n";
        return -1;
    }

    DB::ImgItem item;
    if(-1 == db.get_dcm_item(series_id,item) ) {
        MI_LOG(MI_ERROR) << "get DICOM failed.\n";
        return -1;
    }
    
    const std::string dcm_path = item.dcm_path;
    std::string ai_anno_path = item.annotation_ai_path;
    std::string ai_im_data_path = item.ai_intermediate_data_path;
    bool recal_im_data = false;
    if (ai_im_data_path.empty()) {
        recal_im_data = true;
        ai_im_data_path = dcm_path + "/" + series_id + ".npy";
    }
    if (ai_anno_path.empty()) {
        ai_anno_path = dcm_path + "/" + series_id + ".csv";
    }

    //get python running path
    const std::string pytorch_path = AppConfig::instance()->get_pytorch_path();
    const std::string py_interface_path = AppConfig::instance()->get_py_interface_path();

    std::shared_ptr<medical_ai::AILungEvaulatePyWrapper> wrapper(new medical_ai::AILungEvaulatePyWrapper());
    if(-1 == wrapper->init(pytorch_path.c_str() ,py_interface_path.c_str()) ){
        MI_LOG(MI_ERROR) << "init python env failed.";
        return -1;
    }

    //Preprocess
    if (recal_im_data) {
        char* buffer_im_data = nullptr;
        int buffer_im_data_size = 0;
        clock_t _start = clock();
        if (-1 == wrapper->preprocess(dcm_path.c_str(), buffer_im_data, buffer_im_data_size) ){
            const char* err = wrapper->get_last_err();
            MI_LOG(MI_ERROR) << "AI lung preprocess failed :" << err;
            return -1;
        }
        clock_t _end = clock();
        MI_LOG(MI_DEBUG) << "preprocess cost: " << (double)(_end-_start)/CLOCKS_PER_SEC << " s";

        if (0 != FileUtil::write_raw(ai_im_data_path, buffer_im_data, buffer_im_data_size)) {
            MI_LOG(MI_ERROR) << "write AI intermediate data to: " << ai_im_data_path << " failed.";
            return -1;
        }
    }

    //Evaluate
    medical_ai::AILungEvaulatePyWrapper::VPredictedNodules nodules;
    clock_t _start = clock();
    if(-1 == wrapper->evaluate(ai_im_data_path.c_str(), nodules)) {
        const char* err = wrapper->get_last_err();
        MI_LOG(MI_ERROR) << "evalulate series: " << series_id << " failed: " << err;
        return -1;
    }
    clock_t _end = clock();
    MI_LOG(MI_DEBUG) << "evaluate cost: " << (double)(_end-_start)/CLOCKS_PER_SEC << " s";

    //encode and write to disk
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    for (size_t i=0; i<nodules.size(); ++i) {
        VOISphere voi(Point3(nodules[i].fX,nodules[i].fY,nodules[i].fZ),nodules[i].fRadius*2.0);
        voi.para0 = nodules[i].fProb;
        nodule_set->add_nodule(voi);
    }

    NoduleSetParser ns_parser;
    ns_parser.set_series_id(series_id);
    if(IO_SUCCESS != ns_parser.save_as_csv(ai_anno_path, nodule_set)) {
        MI_LOG(MI_ERROR) << "save evaluated result to " << ai_anno_path << " failed.";
        return -1;
    }

    MI_LOG(MI_INFO) << "evaluate done.";
    return 0;
}