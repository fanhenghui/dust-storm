#ifndef MED_IMG_APPCOMMON_MI_MODEL_DBS_STATUS_H
#define MED_IMG_APPCOMMON_MI_MODEL_DBS_STATUS_H

#include <string>
#include <vector>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelDBSStatus : public IModel {
public:
    ModelDBSStatus();
    virtual ~ ModelDBSStatus() {}

    void reset();

    //success tag
    bool success();
    void set_success();

    //error msg
    void push_error_info(const std::string& err);
    std::vector<std::string> get_error_infos() const;

    //preprocess mask query 
    bool has_preprocess_mask();
    void set_preprocess_mask();

    //ai annotation query
    bool has_ai_annotation();
    void set_ai_annotation();
    //TODO set user annotation

    //after init 
    bool has_init();
    void set_init(); 

    //user trigger ai annotation query but impatient to wait
    bool has_query_ai_annotation();
    void cancel_ai_annotation();
    void query_ai_annotation();

    void wait();
    void unlock();
private:
    bool _success;
    std::vector<std::string> _err_infos;
    bool _has_preprocess_mask;
    bool _has_ai_annotation;
    bool _query_ai_annotation;
    bool _init;

    boost::mutex _mutex;
    boost::condition _condition;
    
private:
    DISALLOW_COPY_AND_ASSIGN(ModelDBSStatus);
};

MED_IMG_END_NAMESPACE
#endif