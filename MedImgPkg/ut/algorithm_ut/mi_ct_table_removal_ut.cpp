#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <stack>

#include "log/mi_logger.h"

#include "util/mi_file_util.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "arithmetic/mi_connected_domain_analysis.h"
#include "arithmetic/mi_segment_threshold.h"
#include "arithmetic/mi_morphology.h"
#include "arithmetic/mi_ct_table_removal.h"
#include "arithmetic/mi_run_length_operator.h"

using namespace medical_imaging;

int ut_ct_table_removal(int argc, char* argv[]) {
    if (argc < 2) {
        return -1;
    }

    const std::string root = argv[1];
    std::vector<std::string> files;
    std::set<std::string> dcm_postfix;
    dcm_postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(root, dcm_postfix, files);

    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> volume_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(files, volume_data, data_header);
    if(status != IO_SUCCESS) {
        MI_LOG(MI_ERROR) << "load file root " << root << " failed.";
        return -1;
    }
    MI_LOG(MI_INFO) << "load DICOM " << data_header->series_uid.c_str() << " done.";
    MI_LOG(MI_INFO) << "dim " << volume_data->_dim[0] << " " << volume_data->_dim[1] << " " << volume_data->_dim[2];
    const unsigned int dim[3] = {volume_data->_dim[0] , volume_data->_dim[1] , volume_data->_dim[2]};
    const unsigned int volume_size = dim[0]*dim[1]*dim[2];
    unsigned char* mask = new unsigned char[volume_size];
    memset(mask, 0, volume_size);

    if (volume_data->_data_type == USHORT) {
        CTTableRemoval<unsigned short> removal;
        removal.set_data_ref((unsigned short*)(volume_data->get_pixel_pointer()));
        removal.set_dim(dim);
        removal.set_mask_ref(mask);
        removal.set_target_label(1);
        removal.set_min_scalar(volume_data->get_min_scalar());
        removal.set_max_scalar(volume_data->get_max_scalar());
        removal.set_image_orientation(volume_data->_image_orientation);
        removal.set_intercept(volume_data->_intercept);
        removal.set_slope(volume_data->_slope);
        removal.remove();
        {
            std::stringstream ss;
            ss << "/home/wangrui22/data/" << data_header->series_uid << "|mask.raw";
            FileUtil::write_raw(ss.str(), mask, volume_size);
        }

        {
            std::stringstream ss;
            ss << "/home/wangrui22/data/" << data_header->series_uid << "|mask.rle";
            std::vector<unsigned int> res = RunLengthOperator::encode(mask, volume_size);
            FileUtil::write_raw(ss.str(), res.data(), res.size()*sizeof(unsigned int));
        }
        
    } else {
        CTTableRemoval<short> removal;
        removal.set_data_ref((short*)(volume_data->get_pixel_pointer()));
        removal.set_dim(dim);
        removal.set_mask_ref(mask);
        removal.set_target_label(1);
        removal.set_min_scalar(volume_data->get_min_scalar());
        removal.set_max_scalar(volume_data->get_max_scalar());
        removal.set_image_orientation(volume_data->_image_orientation);
        removal.set_intercept(volume_data->_intercept);
        removal.set_slope(volume_data->_slope);
        removal.remove();
        {
            std::stringstream ss;
            ss << "/home/wangrui22/data/" << data_header->series_uid << "|mask.raw";
            FileUtil::write_raw(ss.str(), mask, volume_size);
        }

        {
            std::stringstream ss;
            ss << "/home/wangrui22/data/" << data_header->series_uid << "|mask.rle";
            std::vector<unsigned int> res = RunLengthOperator::encode(mask, volume_size);
            FileUtil::write_raw(ss.str(), res.data(), res.size()*sizeof(unsigned int));
        }
        
    }
    MI_LOG(MI_INFO) << "remove table done.";
    return 0;
}