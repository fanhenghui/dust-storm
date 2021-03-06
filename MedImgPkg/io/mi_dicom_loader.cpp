#include "mi_dicom_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmdata/dcdicdir.h"
#include "dcmtk/dcmdata/dcpxitem.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmjpeg/djdecode.h"
#include "dcmtk/dcmdata/dcistrma.h"
#include "dcmtk/dcmdata/dcistrmb.h"

#include "util/mi_model_progress.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

DICOMLoader::DICOMLoader() : _progress(0.0f) {}

DICOMLoader::~DICOMLoader() {}

IOStatus DICOMLoader::check_series_uid(const std::string& file,
                                       std::string& study_uid,
                                       std::string& series_uid) {
    if (file.empty()) {
        return IO_EMPTY_INPUT;
    }

    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->loadFile(file.c_str());

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    OFString context;
    status = data_set->findAndGetOFString(DCM_StudyInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    study_uid = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_SeriesInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    series_uid = std::string(context.c_str());

    return IO_SUCCESS;
}

IOStatus DICOMLoader::check_series_uid(const DCMSliceStream* stream,
                                       std::string& study_uid,
                                       std::string& series_uid) {
    if (nullptr == stream) {
        return IO_EMPTY_INPUT;
    }

    DcmInputBufferStream dcm_buffer_stream;
    dcm_buffer_stream.setBuffer(stream->buffer, stream->size);
    dcm_buffer_stream.setEos();
    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->read(dcm_buffer_stream);

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    OFString context;
    status = data_set->findAndGetOFString(DCM_StudyInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    study_uid = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_SeriesInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    series_uid = std::string(context.c_str());

    return IO_SUCCESS;
}

IOStatus DICOMLoader::check_series_uid(
    const std::string& file, std::string& study_uid, std::string& series_uid,
    std::string& patient_name, std::string& patient_id, std::string& modality) {
    if (file.empty()) {
        return IO_EMPTY_INPUT;
    }

    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->loadFile(file.c_str());

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    OFString context;
    status = data_set->findAndGetOFString(DCM_StudyInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    study_uid = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_SeriesInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    series_uid = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_Modality, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    modality = std::string(context.c_str());

    // name & id may be null
    status = data_set->findAndGetOFString(DCM_PatientName, context);
    patient_name = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_PatientID, context);
    patient_id = std::string(context.c_str());

    return IO_SUCCESS;
}

IOStatus DICOMLoader::check_series_uid(
    const DCMSliceStream* stream, std::string& study_uid, std::string& series_uid,
    std::string& patient_name, std::string& patient_id, std::string& modality) {

    if (nullptr == stream) {
        return IO_EMPTY_INPUT;
    }

    DcmInputBufferStream dcm_buffer_stream;
    dcm_buffer_stream.setBuffer(stream->buffer, stream->size);
    dcm_buffer_stream.setEos();
    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->read(dcm_buffer_stream);

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    OFString context;
    status = data_set->findAndGetOFString(DCM_StudyInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    study_uid = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_SeriesInstanceUID, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    series_uid = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_Modality, context);

    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    modality = std::string(context.c_str());

    // name & id may be null
    status = data_set->findAndGetOFString(DCM_PatientName, context);
    patient_name = std::string(context.c_str());

    status = data_set->findAndGetOFString(DCM_PatientID, context);
    patient_id = std::string(context.c_str());

    return IO_SUCCESS;
}

IOStatus DICOMLoader::get_sop_instance_uid(const std::string& file, std::string& sop_instance_uid) {
    if (file.empty()) {
        return IO_EMPTY_INPUT;
    }

    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->loadFile(file.c_str());

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    OFString context;
    status = data_set->findAndGetOFString(DCM_SOPInstanceUID, context);
    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    sop_instance_uid = context.c_str();

    return IO_SUCCESS;
}

IOStatus DICOMLoader::get_sop_instance_uid(const DCMSliceStream* stream, std::string& sop_instance_uid) {
    if (nullptr == stream) {
        return IO_EMPTY_INPUT;
    }

    DcmInputBufferStream dcm_buffer_stream;
    dcm_buffer_stream.setBuffer(stream->buffer, stream->size);
    dcm_buffer_stream.setEos();
    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->read(dcm_buffer_stream);

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    OFString context;
    status = data_set->findAndGetOFString(DCM_SOPInstanceUID, context);
    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }

    sop_instance_uid = context.c_str();

    return IO_SUCCESS;
}
IOStatus DICOMLoader::get_dicom_info(const std::string& file, 
        PatientInfo& patient_info, StudyInfo& study_info, SeriesInfo& series_info, InstanceInfo& instance_info) {
    if (file.empty()) {
        return IO_EMPTY_INPUT;
    }

    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->loadFile(file.c_str());
    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();
    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    //study uid, series uid, modality, series sop instance uid is not null
    OFString context;
    if (data_set->findAndGetOFString(DCM_StudyInstanceUID, context).bad()) {
        return IO_DATA_DAMAGE;
    }
    study_info.study_uid = std::string(context.c_str());

    if (data_set->findAndGetOFString(DCM_SeriesInstanceUID, context).bad()) {
        return IO_DATA_DAMAGE;
    }
    series_info.series_uid = std::string(context.c_str());

    if (data_set->findAndGetOFString(DCM_Modality, context).bad()) {
        return IO_DATA_DAMAGE;
    }
    series_info.modality = std::string(context.c_str());

    data_set->findAndGetOFString(DCM_SOPInstanceUID, context);
    if (status.bad()) {
        return IO_DATA_DAMAGE;
    }
    instance_info.sop_instance_uid = std::string(context.c_str());


    if (data_set->findAndGetOFString(DCM_PatientName, context).good()) {
        patient_info.patient_name = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_PatientID, context).good()) {
        patient_info.patient_id = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_PatientBirthDate, context).good()) {
        patient_info.patient_birth_date = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_PatientSex, context).good()) {
        patient_info.patient_sex = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_StudyID, context).good()) {
        study_info.study_id = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_AccessionNumber, context).good()) {
        study_info.study_id = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_StudyDescription, context).good()) {
        study_info.study_desc = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_StudyDate, context).good()) {
        study_info.study_date = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_StudyTime, context).good()) {
        study_info.study_time = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_SeriesNumber, context).good()) {
        series_info.series_no = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_InstitutionName, context).good()) {
        series_info.institution = std::string(context.c_str());
    }

    if (data_set->findAndGetOFString(DCM_SeriesDescription, context).good()) {
        series_info.series_desc = std::string(context.c_str());
    }

    if(data_set->findAndGetOFString(DCM_SOPClassUID, context).good()) {
        instance_info.sop_class_uid = std::string(context.c_str());
    }

    return IO_SUCCESS;
}

IO_Export IOStatus DICOMLoader::load_series(std::vector<DCMSliceStream*>& buffers,
                std::shared_ptr<ImageData>& image_data,
                std::shared_ptr<ImageDataHeader>& img_data_header) {
    MI_IO_LOG(MI_TRACE) << "IN load_series.";
    if (buffers.empty()) {
        set_progress(100);
        return IO_EMPTY_INPUT;
    }

    const unsigned int uiSliceCount = static_cast<unsigned int>(buffers.size());
    DcmFileFormatSet data_format_set;

    //////////////////////////////////////////////////////////////////////////
    // 1 load series
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
        DCMSliceStream* slice_buffer = *it;
        DcmInputBufferStream dcm_buffer_stream;
        dcm_buffer_stream.setBuffer(slice_buffer->buffer, slice_buffer->size);
        dcm_buffer_stream.setEos();
        DcmFileFormatPtr file_format(new DcmFileFormat());
        OFCondition status = file_format->read(dcm_buffer_stream);

        if (status.bad()) {
            set_progress(100);
            return IO_FILE_OPEN_FAILED;
        }

        data_format_set.push_back(file_format);
    }

    add_progress(2);

    //TODO add  data check
    //////////////////////////////////////////////////////////////////////////
    // 2 Data check
    // IOStatus checking_status = data_check(files, data_format_set);

    // if (IO_SUCCESS != checking_status) {
    //     set_progress(100);
    //     return checking_status;
    // }

    if (uiSliceCount <
            16) { //不支持少于16张的数据进行三维可视化 // TODO 这一步在这里做不太合适
        set_progress(100);
        return IO_UNSUPPORTED_YET;
    }

    add_progress(3);

    //////////////////////////////////////////////////////////////////////////
    // 3 Sort series
    IOStatus sort_status = sort_series(data_format_set);

    if (IO_SUCCESS != sort_status) {
        set_progress(100);
        return sort_status;
    }

    add_progress(4);

    //////////////////////////////////////////////////////////////////////////
    // 4 Construct image data header
    img_data_header.reset(new ImageDataHeader());
    IOStatus data_heading_status = construct_data_header(data_format_set[0], img_data_header);

    if (IO_SUCCESS != data_heading_status) {
        set_progress(100);
        img_data_header.reset();
        return data_heading_status;
    }

    add_progress(10);

    //////////////////////////////////////////////////////////////////////////
    // 5 Construct image data
    image_data.reset(new ImageData());
    IOStatus data_imaging_status =
        construct_image_data(data_format_set, img_data_header, image_data);

    if (IO_SUCCESS != data_imaging_status) {
        set_progress(100);
        img_data_header.reset();
        image_data.reset();
        return data_imaging_status;
    }

    set_progress(100);
    MI_IO_LOG(MI_TRACE) << "OUT load_series.";
    return IO_SUCCESS;
}

IOStatus DICOMLoader::load_series(std::vector<std::string>& files,
                         std::shared_ptr<ImageData>& image_data,
                         std::shared_ptr<ImageDataHeader>& img_data_header) {
    MI_IO_LOG(MI_TRACE) << "IN load_series.";
    if (files.empty()) {
        set_progress(100);
        return IO_EMPTY_INPUT;
    }

    const unsigned int uiSliceCount = static_cast<unsigned int>(files.size());
    DcmFileFormatSet data_format_set;

    //////////////////////////////////////////////////////////////////////////
    // 1 load series
    for (auto it = files.begin(); it != files.end(); ++it) {
        const std::string file_name = *it;
        DcmFileFormatPtr file_format(new DcmFileFormat());
        OFCondition status = file_format->loadFile(file_name.c_str());

        if (status.bad()) {
            set_progress(100);
            return IO_FILE_OPEN_FAILED;
        }

        data_format_set.push_back(file_format);
    }

    add_progress(2);

    //////////////////////////////////////////////////////////////////////////
    // 2 Data check
    IOStatus checking_status = data_check(files, data_format_set);

    if (IO_SUCCESS != checking_status) {
        set_progress(100);
        return checking_status;
    }

    if (uiSliceCount <
            16) { //不支持少于16张的数据进行三维可视化 // TODO 这一步在这里做不太合适
        set_progress(100);
        return IO_UNSUPPORTED_YET;
    }

    add_progress(3);

    //////////////////////////////////////////////////////////////////////////
    // 3 Sort series
    IOStatus sort_status = sort_series(data_format_set);

    if (IO_SUCCESS != sort_status) {
        set_progress(100);
        return sort_status;
    }

    add_progress(4);

    //////////////////////////////////////////////////////////////////////////
    // 4 Construct image data header
    img_data_header.reset(new ImageDataHeader());
    IOStatus data_heading_status = construct_data_header(data_format_set[0], img_data_header);

    if (IO_SUCCESS != data_heading_status) {
        set_progress(100);
        img_data_header.reset();
        return data_heading_status;
    }

    add_progress(10);

    //////////////////////////////////////////////////////////////////////////
    // 5 Construct image data
    image_data.reset(new ImageData());
    IOStatus data_imaging_status =
        construct_image_data(data_format_set, img_data_header, image_data);

    if (IO_SUCCESS != data_imaging_status) {
        set_progress(100);
        img_data_header.reset();
        image_data.reset();
        return data_imaging_status;
    }

    set_progress(100);
    MI_IO_LOG(MI_TRACE) << "OUT load_series.";
    return IO_SUCCESS;
}

IOStatus DICOMLoader::data_check(std::vector<std::string>& files,
                                   DcmFileFormatSet& file_format_set) {
    IOStatus status = data_check_series_unique(files, file_format_set);
    if (status != IO_SUCCESS) {
        return status;
    }

    DcmFileFormatSet sequece_series_set;
    status = data_check_spacing_unique(file_format_set , sequece_series_set);
    if (status != IO_SUCCESS) {
        return status;
    }
    file_format_set = sequece_series_set;

    return status;
}

IOStatus DICOMLoader::data_check_series_unique(
    std::vector<std::string>& files, DcmFileFormatSet& file_format_set)
{
    std::map<std::string, std::vector<int>> series_separate;
    std::string series_id;
    int idx = 0;

    for (auto it = file_format_set.begin(); it != file_format_set.end();
        ++it, ++idx) {
            OFString context;
            OFCondition status =
                (*it)->getDataset()->findAndGetOFString(DCM_SeriesInstanceUID, context);

            if (status.bad()) {
                return IO_DATA_CHECK_FAILED;
            }

            series_id = std::string(context.c_str());
            auto it_sep = series_separate.find(series_id);

            if (it_sep != series_separate.end()) {
                it_sep->second.push_back(idx);
            } else {
                series_separate[series_id] = std::vector<int>(1, idx);
            }
    }

    if (series_separate.size() != 1) {
        // get majority series
        auto it_sep = series_separate.begin();
        std::string max_num_series_id = it_sep->first;
        size_t max_size = it_sep->second.size();
        ++it_sep;

        for (; it_sep != series_separate.end(); ++it_sep) {
            if (it_sep->second.size() > max_size) {
                max_size = it_sep->second.size();
                max_num_series_id = it_sep->first;
            }
        }

        // get to be delete minority series index
        std::vector<int> to_be_delete;

        for (auto it_sep = series_separate.begin(); it_sep != series_separate.end();
            ++it_sep) {
                if (it_sep->first != max_num_series_id) {
                    to_be_delete.insert(to_be_delete.end(), it_sep->second.begin(),
                        it_sep->second.end());
                }
        }

        std::sort(to_be_delete.begin(), to_be_delete.end(), std::less<int>());

        if (to_be_delete.empty()) {
            return IO_SUCCESS;
        }

        DcmFileFormatSet file_format_set_major;
        std::vector<std::string> file_major;
        file_format_set_major.reserve(file_format_set.size());
        file_major.reserve(file_format_set.size());

        // delete minority series file format
        auto it_delete = to_be_delete.begin();
        int idx_delete = 0;
        auto it_file = files.begin();

        for (auto it = file_format_set.begin(); it != file_format_set.end();
            ++idx_delete, ++it_file, ++it) {
                if(it_delete != to_be_delete.end() && idx_delete == *it_delete) {
                    ++it_delete;
                } else {
                    file_format_set_major.push_back(*it);
                    file_major.push_back(*it_file);
                }
        }

        files = file_major;
        file_format_set = file_format_set_major;
    }

    return IO_SUCCESS;
}

IOStatus DICOMLoader::data_check_spacing_unique(
    DcmFileFormatSet& set_in, DcmFileFormatSet& set_out) {
    std::vector<Point2> spacing(set_in.size());
    for (size_t i = 0; i < spacing.size(); ++i) {
        get_pixel_spacing(set_in[i]->getDataset(), spacing[i]);
    }
    
    std::vector<std::vector<size_t>> same_spacing_begin_end;
    std::vector<size_t> cur_set;
    cur_set.push_back(0);
    same_spacing_begin_end.push_back(cur_set);
    Point2 cur_spacing = spacing[0];
    size_t cur_id = 0;
    for (size_t i = 1 ; i < spacing.size(); ++i) {
        if (spacing[i] != cur_spacing) {
            std::vector<size_t> cur_set;
            cur_set.push_back(i);
            same_spacing_begin_end.push_back(cur_set);
            ++cur_id;
            cur_spacing = spacing[i];
        } else {
            same_spacing_begin_end[cur_id].push_back(i);
        }
    }

    if (cur_id == 0) {
        set_out = set_in;
        return IO_SUCCESS;
    } else {
        //choose major spacing slice to next pipe
        size_t major_slice = same_spacing_begin_end[0].size();
        size_t major_id = 0;
        for (size_t i = 1; i < same_spacing_begin_end.size() ; ++i) {
            if (major_slice < same_spacing_begin_end[i].size()) {
                major_slice = same_spacing_begin_end[i].size();
                major_id = i;
            }
        }

        set_out.resize(major_slice);
        for (size_t i = 0; i < major_slice; ++i) {
            set_out[i] = set_in[same_spacing_begin_end[major_id][i]];
        }
        return IO_SUCCESS;
    }
}

//IOStatus DICOMLoader::data_check_series_image_sequece_i(
//    DcmFileFormatSet& set_in , DcmFileFormatSet& set_out) {
//    sort_series(set_in);
//
//    std::vector<Point3> z_pos(set_in.size());
//    for (size_t i = 0; i < z_pos.size(); ++i) {
//        get_image_position(set_in[i]->getDataset(), z_pos[i]);
//    }
//
//    std::vector<Vector3> z_dir(set_in.size()-1);
//    for (size_t i = 1; i < z_pos.size(); ++i) {
//        z_dir[i-1] = z_pos[i] - z_pos[i-1];
//    }
//    std::vector<std::pair<size_t , size_t>> same_seq_begin_end;
//    same_seq_begin_end.push_back(std::make_pair(0,0));
//    Vector3 cur_dir = z_dir[0];
//    size_t cur_id = 0;
//    size_t major_seq_id = 0;
//    size_t major_seq_slice = 0;
//    for (size_t i = 1 ; i < z_dir.size(); ++i) {
//        if (z_dir[i] != cur_dir) {
//            same_seq_begin_end[cur_id].second = i;
//            size_t cur_slice = i - same_seq_begin_end[cur_id].first + 1;
//            if (cur_slice > major_seq_slice) {
//                major_seq_slice = cur_slice;
//                major_seq_id = cur_id;
//            }
//            same_seq_begin_end.push_back(std::make_pair(i+1,i+1));
//            ++cur_id;
//        }
//    }
//    if (same_seq_begin_end[cur_id].second == same_seq_begin_end[cur_id].first) {
//        same_seq_begin_end[cur_id].second = z_dir.size();
//    }
//
//    if(cur_id == 0) {
//        set_out = set_in;
//        return IO_SUCCESS;
//    } else {
//        set_out.resize(major_seq_slice);
//        int idx = 0;
//        for (size_t i = same_seq_begin_end[major_seq_id].first; i <= same_seq_begin_end[major_seq_id].second; ++i) {
//            set_out[idx++] = set_in[i];
//        }
//        return IO_SUCCESS;
//    }
//}

IOStatus DICOMLoader::sort_series(DcmFileFormatSet& file_format_set) {
    // sort based on slice lotation
    std::map<double, int> locs;

    for (size_t i = 0; i < file_format_set.size(); ++i) {
        DcmDataset* data_set = file_format_set[i]->getDataset();

        if (!data_set) {
            return IO_DATA_DAMAGE;
        }

        double sl(0);
        get_slice_location(data_set, sl);

        if (locs.find(sl) != locs.end()) {
            // Same slice location
            return IO_DATA_DAMAGE;
        }

        locs[sl] = (int)i;
    }

    DcmFileFormatSet new_set(file_format_set.size());
    int idx = 0;

    for (auto it = locs.begin(); it != locs.end(); ++it) {
        new_set[idx++] = file_format_set[it->second];
    }

    DcmDataset* data_set = new_set[0]->getDataset();
    DcmDataset* data_set_last = new_set[new_set.size() - 1]->getDataset();
    Point3 pt_first, pt_last;
    get_image_position(data_set, pt_first);
    get_image_position(data_set_last, pt_last);

    if (pt_first.z <
            pt_last.z) { // image postion is the same sequence with slice location
        file_format_set = std::move(new_set);
    } else {
        DcmFileFormatSet new_set2(file_format_set.size());

        for (size_t i = 0; i < new_set.size(); ++i) {
            new_set2[i] = new_set[new_set.size() - i - 1];
        }

        file_format_set = std::move(new_set2);
    }

    return IO_SUCCESS;
}

IOStatus DICOMLoader::construct_data_header(
    DcmFileFormatPtr& file_format,
    std::shared_ptr<ImageDataHeader> img_data_header) {
    IOStatus io_status = IO_DATA_DAMAGE;

    try {
        DcmDataset* data_set = file_format->getDataset();
        if (!data_set) {
            io_status = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Get data set failed!");
        }

        DcmMetaInfo* meta_info = file_format->getMetaInfo();
        if (!meta_info) {
            io_status = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Get meta infp failed!");
        }

        std::string err_info = "";

        // 4.0 Get SOP Instance UID 
        if (!get_sop_instance_uid(data_set, img_data_header)) {
            err_info += "Parse tag SOPInstanceUID failed! ";
        }

        // 4.1 Get Transfer Syntax UID
        if (!get_transfer_syntax_uid(meta_info, img_data_header)) {
            err_info += "Parse tag TransferSyntaxUID failed! ";
        }

        // 4.2 Get Study UID
        if (!get_study_uid(data_set, img_data_header)) {
            err_info += "Parse tag StudyUID failed! ";
        }

        // 4.3 Get Series UID
        if (!get_series_uid(data_set, img_data_header)) {
            err_info += "Parse tag SeriesUID failed! ";
        }

        // 4.4 Get Date
        if (!get_content_time(data_set, img_data_header)) {
            // io_status = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag ContentTime failed!");
        }

        // 4.5 Get Modality
        OFString modality;
        OFCondition status = data_set->findAndGetOFString(DCM_Modality, modality);

        if (status.bad()) {
            err_info += "Parse tag Modality failed! ";
        }
        img_data_header->modality = modality_to_enum(modality.c_str());

        // 4.6 Manufacturer
        if (!get_manufacturer(data_set, img_data_header)) {
            // io_status = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag Manufacturer failed!");
        }

        // 4.7 Manufacturer model
        if (!get_manufacturer_model_name(data_set, img_data_header)) {
            // io_status = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag ManufacturerModelName failed!");
        }

        // 4.8 Patient name
        if (!get_patient_name(data_set, img_data_header)) {
            // io_status = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag PatientName failed!");
        }

        // 4.9 Patient ID
        if (!get_patient_id(data_set, img_data_header)) {
            // io_status = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag PatientID failed!");
        }

        // 4.10 Patient Sex(很多图像都没有这个Tag)
        if (!get_patient_sex(data_set, img_data_header)) {
            // eLoadingStatus = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag PatientSex failed!");
        }

        // 4.11 Patient Age(很多图像都没有这个Tag)
        if (!get_patient_age(data_set, img_data_header)) {
            /*eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientAge failed!");*/
        }

        // 4.12 Slice thickness (不是一定必要)
        if (!get_slice_thickness(data_set, img_data_header)) {
            // eLoadingStatus = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag SliceThickness failed!");
        }

        // 4.13 KVP (CT only)
        if (!get_kvp(data_set, img_data_header)) {
            // eLoadingStatus = IO_DATA_DAMAGE;
            // IO_THROW_EXCEPTION("Parse tag KVP failed!");
        }

        // 4.14 Patient position
        if (!get_patient_position(data_set, img_data_header)) {
            err_info += "Parse tag PatientPositio failed! ";
        }

        // 4.15 Samples per Pixel
        if (!get_sample_per_pixel(data_set, img_data_header)) {
            err_info += "Parse tag SamplePerPixel failed! ";
        }

        // 4.16 Photometric Interpretation
        OFString pi;
        status = data_set->findAndGetOFString(DCM_PhotometricInterpretation, pi);

        if (status.bad()) {
            err_info += "Parse tag PhotometricInterpretation failed! ";
        }

        const std::string pi_std = std::string(pi.c_str());

        if ("MONOCHROME1" == pi_std) {
            img_data_header->photometric_interpretation = PI_MONOCHROME1;
        } else if ("MONOCHROME2" == pi_std) {
            img_data_header->photometric_interpretation = PI_MONOCHROME2;
        } else if ("RGB" == pi_std) {
            img_data_header->photometric_interpretation = PI_RGB;
        } else {
            err_info += "Unsupport photometric Interpretation YET! ";
        }

        // 4.17 Rows
        if (!get_rows(data_set, img_data_header)) {
            err_info += "Parse tag Rows failed! ";
        }

        // 4.18 Columns
        if (!get_columns(data_set, img_data_header)) {
            err_info += "Parse tag Columns failed! ";
        }

        // 4.19 Pixel Spacing
        if (!get_pixel_spacing(data_set, img_data_header)) {
            err_info += "Parse tag PixelSpacing failed! ";
        }

        // 4.20 Bits Allocated
        if (!get_bits_allocated(data_set, img_data_header)) {
            err_info += "Parse tag BitsAllocated failed! ";
        }

        // 4.21 Pixel Representation
        if (!get_pixel_representation(data_set, img_data_header)) {
            err_info += "Parse tag PixelRepresentation failed! ";
        }
        
        if (!err_info.empty()) {
            io_status = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION(err_info);
        }

        return IO_SUCCESS;
    } catch (const Exception& e) {
        MI_IO_LOG(MI_FATAL) << "construct header failed : " << e.what();
        return io_status;
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_FATAL) << "construct header failed : " << e.what();
        return io_status;
    }
}

IOStatus DICOMLoader::construct_image_data(
    DcmFileFormatSet& file_format_set,
    std::shared_ptr<ImageDataHeader> data_header,
    std::shared_ptr<ImageData> image_data) {
    const unsigned int slice_count =
        static_cast<unsigned int>(file_format_set.size());
    DcmFileFormatPtr file_format_first = file_format_set[0];
    DcmDataset* data_set = file_format_first->getDataset();

    // Intercept and slope
    get_intercept(data_set, image_data->_intercept);
    get_slope(data_set, image_data->_slope);

    data_header->slice_location.resize(slice_count);
    data_header->image_position.resize(slice_count);

    for (unsigned int i = 0; i < slice_count; ++i) {
        double slice_location = 0;
        Point3 image_position;
        DcmDataset* dataset = file_format_set[i]->getDataset();

        if (!dataset) {
            return IO_DATA_DAMAGE;
        }

        get_slice_location(dataset, slice_location);
        data_header->slice_location[i] = slice_location;

        get_image_position(dataset, image_position);
        data_header->image_position[i] = image_position;
    }

    // Data channel
    if (PI_RGB == data_header->photometric_interpretation &&
            3 == data_header->sample_per_pixel) {
        image_data->_channel_num = 3;
    } else if ((PI_MONOCHROME1 == data_header->photometric_interpretation ||
                PI_MONOCHROME2 == data_header->photometric_interpretation) &&
               1 == data_header->sample_per_pixel) {
        image_data->_channel_num = 1;
    } else {
        return IO_UNSUPPORTED_YET;
    }

    // Data type
    unsigned int image_size = data_header->rows * data_header->columns;

    if (8 == data_header->bits_allocated) {
        if (0 == data_header->pixel_representation) {
            image_data->_data_type = UCHAR;
        } else {
            image_data->_data_type = CHAR;
        }
    } else if (16 == data_header->bits_allocated) {
        image_size *= 2;

        if (0 == data_header->pixel_representation) {
            image_data->_data_type = USHORT;
        } else {
            image_data->_data_type = SHORT;
        }
    } else {
        return IO_UNSUPPORTED_YET;
    }

    // Dimension
    image_data->_dim[0] = data_header->columns;
    image_data->_dim[1] = data_header->rows;
    image_data->_dim[2] = slice_count;

    // Spacing
    image_data->_spacing[0] = data_header->pixel_spacing[1];
    image_data->_spacing[1] = data_header->pixel_spacing[0];
    const double slice_location_first = data_header->slice_location[0];
    const double slice_location_last =
        data_header->slice_location[slice_count - 1];
    image_data->_spacing[2] = fabs((slice_location_last - slice_location_first) /
                                   static_cast<double>(slice_count - 1));

    // Image position in patient
    image_data->_image_position = data_header->image_position[0];

    // Image Orientation in patient
    Vector3 row_orientation;
    Vector3 column_orientation;

    if (!get_image_orientation(data_set, row_orientation,
                                 column_orientation)) {
        return IO_DATA_DAMAGE;
    }

    image_data->_image_orientation[0] = row_orientation;
    image_data->_image_orientation[1] = column_orientation;
    image_data->_image_orientation[2] =
        data_header->image_position[slice_count - 1] -
        data_header->image_position[0];
    image_data->_image_orientation[2].normalize();

    // Image data
    image_data->mem_allocate();
    char* data_array = (char*)(image_data->get_pixel_pointer());
    // DICOM transfer syntaxes
    const std::string TSU_LittleEndianImplicitTransferSyntax =
        std::string("1.2.840.10008.1.2"); // Default transfer for DICOM
    const std::string TSU_LittleEndianExplicitTransferSyntax =
        std::string("1.2.840.10008.1.2.1");
    const std::string TSU_DeflatedExplicitVRLittleEndianTransferSyntax =
        std::string("1.2.840.10008.1.2.1.99");
    const std::string TSU_BigEndianExplicitTransferSyntax =
        std::string("1.2.840.10008.1.2.2");

    // JEPG Lossless
    const std::string TSU_JPEGProcess14SV1TransferSyntax =
        std::string("1.2.840.10008.1.2.4.70"); // Default Transfer Syntax for
    // Lossless JPEG Image Compression
    const std::string TSU_JPEGProcess14TransferSyntax =
        std::string("1.2.840.10008.1.2.4.57");

    // JEPG2000 需要购买商业版的 dcmtk
    const std::string TSU_JEPG2000CompressionLosslessOnly =
        std::string("1.2.840.10008.1.2.4.90");
    const std::string TSU_JEPG2000Compression =
        std::string("1.2.840.10008.1.2.4.91");

    const std::string& my_tsu = data_header->transfer_syntax_uid;

    const float progress_step = 1.0f / static_cast<float>(slice_count) * 90.0f;

    if (my_tsu == TSU_LittleEndianImplicitTransferSyntax ||
            my_tsu == TSU_LittleEndianExplicitTransferSyntax ||
            my_tsu == TSU_DeflatedExplicitVRLittleEndianTransferSyntax ||
            my_tsu == TSU_BigEndianExplicitTransferSyntax) {
        for (unsigned int i = 0; i < slice_count; ++i) {
            DcmDataset* dataset = file_format_set[i]->getDataset();

            if (!dataset) {
                return IO_DATA_DAMAGE;
            }

            get_pixel_data(file_format_set[i], dataset, data_array + image_size * i,
                             image_size);

            add_progress(progress_step);
        }
    } else if (my_tsu == TSU_JPEGProcess14SV1TransferSyntax ||
               my_tsu == TSU_JPEGProcess14TransferSyntax) {
        for (unsigned int i = 0; i < slice_count; ++i) {
            DcmDataset* dataset = file_format_set[i]->getDataset();

            if (!dataset) {
                return IO_DATA_DAMAGE;
            }

            get_jpeg_compressed_pixel_data(file_format_set[i], dataset,
                                             data_array + image_size * i, image_size);

            add_progress(progress_step);
        }
    } else if (my_tsu == TSU_JEPG2000CompressionLosslessOnly ||
               my_tsu == TSU_JEPG2000Compression) {
        return IO_UNSUPPORTED_YET;
    } else {
        return IO_UNSUPPORTED_YET;
    }

    return IO_SUCCESS;
}

bool DICOMLoader::get_transfer_syntax_uid(
    DcmMetaInfo* meta_info, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        meta_info->findAndGetOFString(DCM_TransferSyntaxUID, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->transfer_syntax_uid = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_content_time(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_ContentDate, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->image_date = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_manufacturer(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_Manufacturer, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->manufacturer = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_manufacturer_model_name(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_ManufacturerModelName, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->manufacturer_model_name = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_patient_name(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientName, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->patient_name = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_patient_id(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientID, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->patient_id = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_patient_sex(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientSex, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->patient_sex = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_patient_age(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientAge, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->patient_age = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_slice_thickness(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_SliceThickness, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->slice_thickness = static_cast<double>(atof(context.c_str()));
    return true;
}

bool DICOMLoader::get_kvp(DcmDataset* data_set,
                            std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_KVP, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->kvp = static_cast<float>(atof(context.c_str()));
    return true;
}

bool DICOMLoader::get_patient_position(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_PatientPosition, context);

    if (status.bad()) {
        return false;
    }

    const std::string patient_position = std::string(context.c_str());

    if ("HFP" == patient_position) {
        img_data_header->patient_position = HFP;
    } else if ("HFS" == patient_position) {
        img_data_header->patient_position = HFS;
    } else if ("HFDR" == patient_position) {
        img_data_header->patient_position = HFDR;
    } else if ("HFDL" == patient_position) {
        img_data_header->patient_position = HFDL;
    } else if ("FFP" == patient_position) {
        img_data_header->patient_position = FFP;
    } else if ("FFS" == patient_position) {
        img_data_header->patient_position = FFS;
    } else if ("FFDR" == patient_position) {
        img_data_header->patient_position = FFDR;
    } else if ("FFDL" == patient_position) {
        img_data_header->patient_position = FFDL;
    }

    return true;
}

bool DICOMLoader::get_series_uid(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_SeriesInstanceUID, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->series_uid = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_study_uid(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_StudyInstanceUID, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->study_uid = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_sop_instance_uid(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_SOPInstanceUID, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->sop_instance_uid = std::string(context.c_str());
    return true;
}

bool DICOMLoader::get_sample_per_pixel(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    unsigned short context = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_SamplesPerPixel, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->sample_per_pixel = static_cast<unsigned int>(context);
    return true;
}

bool DICOMLoader::get_rows(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    unsigned short context = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_Rows, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->rows = static_cast<unsigned int>(context);
    return true;
}

bool DICOMLoader::get_columns(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    unsigned short context = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_Columns, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->columns = static_cast<unsigned int>(context);
    return true;
}

bool DICOMLoader::get_pixel_spacing(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    OFString row_spacing, col_spacing;
    OFCondition status1 =
        data_set->findAndGetOFString(DCM_PixelSpacing, row_spacing, 0);
    OFCondition status2 =
        data_set->findAndGetOFString(DCM_PixelSpacing, col_spacing, 1);

    if (status1.bad() || status2.bad()) {
        return false;
    }

    img_data_header->pixel_spacing[0] = atof(row_spacing.c_str());
    img_data_header->pixel_spacing[1] = atof(col_spacing.c_str());

    return true;
}

bool DICOMLoader::get_pixel_spacing(
    DcmDataset* data_set, Point2& spacing) {
    OFString row_spacing, col_spacing;
    OFCondition status1 =
        data_set->findAndGetOFString(DCM_PixelSpacing, row_spacing, 0);
    OFCondition status2 =
        data_set->findAndGetOFString(DCM_PixelSpacing, col_spacing, 1);

    if (status1.bad() || status2.bad()) {
        return false;
    }

    spacing.x = atof(row_spacing.c_str());
    spacing.y = atof(col_spacing.c_str());
    return true;
}

bool DICOMLoader::get_bits_allocated(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    unsigned short context = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_BitsAllocated, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->bits_allocated = static_cast<unsigned int>(context);
    return true;
}

bool DICOMLoader::get_pixel_representation(
    DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header) {
    unsigned short context = 0;
    OFCondition status =
        data_set->findAndGetUint16(DCM_PixelRepresentation, context);

    if (status.bad()) {
        return false;
    }

    img_data_header->pixel_representation = static_cast<unsigned int>(context);
    return true;
}

bool DICOMLoader::get_intercept(DcmDataset* data_set, float& intercept) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_RescaleIntercept, context);

    if (status.bad()) {
        return false;
    }

    intercept = (float)atof(context.c_str());
    return true;
}

bool DICOMLoader::get_slope(DcmDataset* data_set, float& slope) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_RescaleSlope, context);

    if (status.bad()) {
        return false;
    }

    slope = (float)atof(context.c_str());
    return true;
}

bool DICOMLoader::get_instance_number(DcmDataset* data_set,
                                        int& instance_num) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_InstanceNumber, context);

    if (status.bad()) {
        return false;
    }

    instance_num = atoi(context.c_str());
    return true;
}

bool DICOMLoader::get_image_position(DcmDataset* data_set,
                                       Point3& image_position) {
    OFString context;
    OFCondition status =
        data_set->findAndGetOFString(DCM_ImagePositionPatient, context, 0);

    if (status.bad()) {
        return false;
    }

    image_position.x = static_cast<double>(atof(context.c_str()));

    status = data_set->findAndGetOFString(DCM_ImagePositionPatient, context, 1);

    if (status.bad()) {
        return false;
    }

    image_position.y = static_cast<double>(atof(context.c_str()));

    status = data_set->findAndGetOFString(DCM_ImagePositionPatient, context, 2);

    if (status.bad()) {
        return false;
    }

    image_position.z = static_cast<double>(atof(context.c_str()));

    return true;
}

bool DICOMLoader::get_image_orientation(DcmDataset* data_set,
        Vector3& row_orientation,
        Vector3& column_orientation) {
    double img_orientation[6] = {0};

    for (int i = 0; i < 6; ++i) {
        OFString context;
        OFCondition status =
            data_set->findAndGetOFString(DCM_ImageOrientationPatient, context, i);

        if (status.bad()) {
            return false;
        }

        img_orientation[i] = static_cast<double>(atof(context.c_str()));
    }

    row_orientation =
        Vector3(img_orientation[0], img_orientation[1], img_orientation[2]);
    column_orientation =
        Vector3(img_orientation[3], img_orientation[4], img_orientation[5]);

    return true;
}

bool DICOMLoader::get_slice_location(DcmDataset* data_set,
                                       double& slice_location) {
    OFString context;
    OFCondition status = data_set->findAndGetOFString(DCM_SliceLocation, context);

    if (status.bad()) {
        return false;
    }

    slice_location = atof(context.c_str());
    return true;
}

bool DICOMLoader::get_pixel_data(DcmFileFormatPtr pFileFormat,
                                   DcmDataset* data_set, char* data_array,
                                   unsigned int length) {
    const unsigned char* dara_ref;
    OFCondition status = data_set->findAndGetUint8Array(DCM_PixelData, dara_ref);

    if (status.bad()) {
        return false;
    }

    memcpy(data_array, dara_ref, length);

    return true;
}

bool DICOMLoader::get_jpeg_compressed_pixel_data(DcmFileFormatPtr file_format,
        DcmDataset* data_set,
        char* data_array,
        unsigned int length) {
    // Code from : http://support.dcmtk.org/docs/mod_dcmjpeg.html
    // Write to a temp decompressed file , then read the decompressed one

    DJDecoderRegistration::registerCodecs(); // register JPEG codecs

    // decompress data set if compressed
    data_set->chooseRepresentation(EXS_LittleEndianExplicit, NULL);

    // check if everything went well
    if (data_set->canWriteXfer(EXS_LittleEndianExplicit)) {
        file_format->saveFile("test_decompressed.dcm", EXS_LittleEndianExplicit);
    }

    DJDecoderRegistration::cleanup(); // deregister JPEG codecs

    file_format->loadFile("test_decompressed.dcm");
    DcmDataset* dataset = file_format->getDataset();

    return get_pixel_data(file_format, dataset, data_array, length);
}

void DICOMLoader::set_progress_model(std::shared_ptr<ProgressModel> model) {
    _model = model;
}

void DICOMLoader::add_progress(float value) {
    if (_model) {
        _progress += value;
        int progress = static_cast<int>(_progress);
        progress = progress > 100 ? 100 : progress;
        _model->set_progress(progress);
        _model->notify();
    }
}

void DICOMLoader::set_progress(int value) {
    if (_model) {
        _progress = static_cast<float>(value);
        _model->set_progress(value);
        _model->notify();
    }
}

IOStatus DICOMLoader::load_dicom(std::string& file, std::shared_ptr<ImageDataHeader>& img_data_header) {
    if (file.empty()) {
        return IO_EMPTY_INPUT;
    }

    DcmFileFormatPtr file_format(new DcmFileFormat());
    OFCondition status = file_format->loadFile(file.c_str());

    if (status.bad()) {
        return IO_FILE_OPEN_FAILED;
    }

    DcmDataset* data_set = file_format->getDataset();

    if (nullptr == data_set) {
        return IO_FILE_OPEN_FAILED;
    }

    img_data_header.reset(new ImageDataHeader());
    IOStatus data_heading_status = construct_data_header(file_format, img_data_header);

    if (IO_SUCCESS != data_heading_status) {
        set_progress(100);
        img_data_header.reset();
        return data_heading_status;
    }

    return IO_SUCCESS;
}

MED_IMG_END_NAMESPACE