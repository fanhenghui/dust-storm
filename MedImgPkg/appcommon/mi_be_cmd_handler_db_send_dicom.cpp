#include "mi_be_cmd_handler_db_send_dicom.h"

#ifdef WIN32
#else 
#include <sys/stat.h>//for create direction
#endif

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_file_util.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "io/mi_message.pb.h"
#include "io/mi_configure.h"
#include "io/mi_cache_db.h"

#include "renderalgo/mi_volume_infos.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_logger.h"
#include "mi_app_common_util.h"
#include "mi_model_dbs_status.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerDBSendDICOM::BECmdHandlerDBSendDICOM(std::shared_ptr<AppController> controller):
    _controller(controller),_th_running(false),_cur_recv_slice(0),_cur_save_slice(0),_total_slice(0),
    _err_tag(false),_series_id(""),_study_id(""),_series_path(""),
    _patient_name(""),_patient_id(""),_modality(""),_size_mb(0) {

}

BECmdHandlerDBSendDICOM::~BECmdHandlerDBSendDICOM() {
    _th_cache_db.join();
    //remove package
    for (auto it = _dcm_streams_store.begin(); it != _dcm_streams_store.end(); ++it) {
        DICOMLoader::DCMSliceStream* stream = *it;
        delete stream;
        stream = nullptr;
    }
    _dcm_streams_store.clear();
}

int BECmdHandlerDBSendDICOM::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    try {
        MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerDBSendDICOM.";
        
        APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
        std::shared_ptr<AppController> controller = _controller.lock();
        APPCOMMON_CHECK_NULL_EXCEPTION(controller);
        std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
        APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);

        const unsigned int end_tag = ipcheader.reserved0;
        const int dcm_slice_sum = (int)ipcheader.reserved1;

        //trigger a thread to write to cache DB
        if (!_th_running) {
            _th_cache_db = boost::thread(boost::bind(&BECmdHandlerDBSendDICOM::update_cache_db, this));
            _th_running = true;
            _cur_recv_slice = 0;
            _total_slice = dcm_slice_sum;
        }

        if (dcm_slice_sum == 0 || dcm_slice_sum != _total_slice) {
            _err_tag = true;
            model_dbs_status->push_error_info("dcm slice sum is 0 when recv dbs DICOM series stream.");
            _condition_cache_db.notify_one();
            return -1;
        }

        //gather the stream
        DICOMLoader::DCMSliceStream* new_stream = new DICOMLoader::DCMSliceStream(buffer,ipcheader.data_len);
        _dcm_streams_store.push_back(new_stream);

        //push new stream into queue
        {
            boost::mutex::scoped_lock locker(_mutex_cache_db);
            _dcm_streams_queue.push_back(new_stream);
            _condition_cache_db.notify_one();
        }    

        ++_cur_recv_slice;
        if (end_tag == 1 || _cur_recv_slice == dcm_slice_sum) {
            if (!(end_tag == 1 && _cur_recv_slice == dcm_slice_sum)) {
                _err_tag = true;
                return -1;
            } else {
                //load DICOM streams
                std::shared_ptr<ImageDataHeader> data_header;
                std::shared_ptr<ImageData> img_data;
                DICOMLoader loader;
                IOStatus status = loader.load_series(_dcm_streams_store, img_data, data_header);
                if (status != IO_SUCCESS) {
                    MI_APPCOMMON_LOG(MI_FATAL) << "load DICOM series stream failed.";
                    return -1;
                }
            
                // create volume infos
                const GPUPlatform gpu_platform = Configure::instance()->get_gpu_platform_type() == CUDA ? CUDA_BASE : GL_BASE;
                std::shared_ptr<VolumeInfos> volume_infos(new VolumeInfos(GPU_BASE, gpu_platform));
                volume_infos->set_data_header(data_header);
                volume_infos->set_volume(img_data); // load volume texture if has graphic card
            
                // create empty mask
                std::shared_ptr<ImageData> mask_data(new ImageData());
                img_data->shallow_copy(mask_data.get());
                mask_data->_channel_num = 1;
                mask_data->_data_type = medical_imaging::UCHAR;
                mask_data->mem_allocate();
                volume_infos->set_mask(mask_data);
            
                controller->set_volume_infos(volume_infos);

                MI_APPCOMMON_LOG(MI_INFO) << "load DBS DICOM stream success.";
            }
        }
        MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerDBSendDICOM.";
        return 0;
    } catch(const Exception& e) {
        return -1;
    }
}

void BECmdHandlerDBSendDICOM::update_cache_db() {
    //TODO 这里的代码之后交给Cache dispatcher来做（单机部署不需要）

    while(true) {
        DICOMLoader::DCMSliceStream *dcm_stream = nullptr;
        {
            //pop slice buffer to write to local disk
            boost::mutex::scoped_lock locker(_mutex_cache_db);
            while(_dcm_streams_queue.empty()){
                _condition_cache_db.wait(_mutex_cache_db);
            }

            dcm_stream = _dcm_streams_queue.front();
            _dcm_streams_queue.pop_front();
        }
        
        if (nullptr == dcm_stream) {
            MI_APPCOMMON_LOG(MI_ERROR) << "DCM stream is null when try to save to cache DB.";
            break;
        }

        if (_err_tag) {
            break;
        }

        if (0 == _cur_save_slice) {
            _dcm_instance_infos.clear();

            //check study and series
            DICOMLoader loader;
            if (IO_SUCCESS != loader.check_series_uid(dcm_stream, _study_id, _series_id, 
                _patient_name, _patient_id, _modality)) {
                MI_APPCOMMON_LOG(MI_ERROR) << "check DCM stream failed when try to save to cahce DB.";
                break;
            }

            //get cache DB path
            std::string ip_port,user,pwd,db_name,path;
            Configure::instance()->get_cache_db_info(ip_port,user,pwd,db_name,path);
            if (path == "") {
                MI_APPCOMMON_LOG(MI_ERROR) << "DB cache is empty when try to save to cache DB.";
                break;
            }

            //create direction
            const std::string study_path = path + "/" + _study_id;
            const std::string series_path = path + "/" + _study_id + "/" +_series_id;
            _series_path = series_path;
#ifdef WIN32
            //TODO check and create direction use windows os function
#else
            if (-1 == access(study_path.c_str(), F_OK)) {
                if (-1 == mkdir(study_path.c_str(), S_IRWXU) ) {
                    MI_APPCOMMON_LOG(MI_ERROR) << "create study direction when try to save to cache DB.";
                    break;
                }
            }
            if (-1 == access(series_path.c_str(), F_OK)) {
                if (-1 ==mkdir(series_path.c_str(), S_IRWXU)) {
                    MI_APPCOMMON_LOG(MI_ERROR) << "create series direction when try to save to cache DB.";
                    break;
                }
            }
#endif
        }

        //save to path
        //get file name
        DICOMLoader loader;
        std::string sop_instance_uid;
        if (IO_SUCCESS != loader.get_sop_instance_uid(dcm_stream, sop_instance_uid)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "check DCM stream failed when try to save to cahce DB.";
            break;
        }

        std::stringstream ss;
        ss << _series_path << "/" <<_modality << "." << sop_instance_uid << ".dcm";
        if(0 != FileUtil::write_raw(ss.str(), dcm_stream->buffer, dcm_stream->size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "write to disk failed when try to save to cache DB.";
            break;
        }
        
        int64_t file_size = 0;
        if (0 != FileUtil::get_file_size(ss.str(), file_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "get instance file size failed when try to save to cache DB.";
            break;
        }

        _dcm_instance_infos.push_back(InstanceInfo());
        InstanceInfo& instance_info = _dcm_instance_infos[_dcm_instance_infos.size()-1];
        instance_info.file_path = ss.str();
        instance_info.file_size = file_size;
        instance_info.sop_instance_uid = sop_instance_uid;

        ++_cur_save_slice;
        _size_mb +=  dcm_stream->size/1024.0/1024.0;

        //add a column to DB
        if (_cur_save_slice == _total_slice) {
            std::string ip_port,user,pwd,db_name,path;
            Configure::instance()->get_cache_db_info(ip_port,user,pwd,db_name,path);
            CacheDB cache_db;
            if (0 != cache_db.connect(user,ip_port,pwd,db_name)) {
                MI_APPCOMMON_LOG(MI_ERROR) << "connect cache DB failed.";
                break;
            }

            if(0 != cache_db.insert_series(_series_id, _dcm_instance_infos)){
                MI_APPCOMMON_LOG(MI_ERROR) << "insert item to cache DB failed.";
            }

            _dcm_instance_infos.clear();

            MI_APPCOMMON_LOG(MI_INFO) << "insert new DICOM item: " << _series_id << " to cache DB sucess.";

            //TODO check Cache DB memory status, and remove oldest cache
            break;
        }
    }
}

MED_IMG_END_NAMESPACE
