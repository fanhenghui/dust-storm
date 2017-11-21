#include "mi_dicom_rt_struct_loader.h"
#include "mi_dicom_rt_struct.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"

#include "boost/algorithm/string.hpp"

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmdata/dcdicdir.h"
#include "dcmtk/dcmdata/dcpxitem.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmjpeg/djdecode.h"

#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

DICOMRTLoader::DICOMRTLoader() {}

DICOMRTLoader::~DICOMRTLoader() {}

IOStatus DICOMRTLoader::load_rt_struct(const std::string& file_name,
                                       std::shared_ptr<RTStruct>& rt_struct) {
    try {
        MI_IO_LOG(MI_TRACE) << "IN load RT struct: " << file_name;
        DcmFileFormatPtr file_format(new DcmFileFormat());
        OFCondition status = file_format->loadFile(file_name.c_str());

        if (status.bad()) {
            MI_LOG(MI_ERROR) << "load rt struct: " << file_name << "failed.";
            return IO_FILE_OPEN_FAILED;
        }

        IOStatus io_status = load_rt_struct(file_format, rt_struct);

        MI_IO_LOG(MI_TRACE) << "OUT load RT struct: " << file_name;
        return io_status;

    } catch (const Exception& e) {
        MI_LOG(MI_FATAL) << "load RT struct file: " << file_name << "failed with exception: " << e.what();
        return IO_FILE_OPEN_FAILED;
    }
}

IOStatus DICOMRTLoader::load_rt_struct(DcmFileFormatPtr file_format,
                                       std::shared_ptr<RTStruct>& rt_struct) {
    try {
        rt_struct.reset(new RTStruct());

        DcmDataset* data_set = file_format->getDataset();

        // 1 Get modality
        OFString modality;
        OFCondition status = data_set->findAndGetOFString(DCM_Modality, modality);

        if (status.bad()) {
            MI_LOG(MI_ERROR) << "get modality failed.";
            return IO_FILE_OPEN_FAILED;
        }

        if ("RTSTRUCT" != modality) {
            MI_LOG(MI_ERROR) << "modality is not RTSTRUCT.";
            return IO_DATA_DAMAGE;
        }

        MI_LOG(MI_DEBUG) << "modality is : " << modality.c_str();

        DcmSequenceOfItems* roi_sequence = nullptr;
        status = data_set->findAndGetSequence(DCM_StructureSetROISequence, roi_sequence);

        if (status.bad()) {
            MI_LOG(MI_ERROR) << "get ROI set sequence failed.";
            return IO_DATA_DAMAGE;
        }

        DcmSequenceOfItems* contour_sequence = nullptr;
        status = data_set->findAndGetSequence(DCM_ROIContourSequence, contour_sequence);

        if (status.bad()) {
            MI_LOG(MI_ERROR) << "get ROI contour sequence failed.";
            return IO_DATA_DAMAGE;
        }

        const unsigned long roi_num = roi_sequence->card();
        const unsigned long contour_num = contour_sequence->card();

        if (roi_num != contour_num) {
            MI_LOG(MI_ERROR) << "ROI num is not match with contour num.";
            return IO_DATA_DAMAGE;
        }

        for (unsigned long i = 0; i < roi_num; ++i) {
            DcmItem* roi_item = roi_sequence->getItem(i);
            DcmItem* coutour_item = contour_sequence->getItem(i);

            OFString roi_name;
            status = roi_item->findAndGetOFString(DCM_ROIName, roi_name);

            if (status.bad()) {
                MI_LOG(MI_ERROR) << "get ROI name failed.";
                return IO_DATA_DAMAGE;
            }

            DcmSequenceOfItems* contour_unit_sequence = nullptr;
            status = coutour_item->findAndGetSequence(DCM_ContourSequence,
                     contour_unit_sequence);

            if (status.bad()) {
                MI_LOG(MI_ERROR) << "get contour sequence failed.";
                return IO_DATA_DAMAGE;
            }

            unsigned long contour_unit_num = contour_unit_sequence->card();

            for (unsigned long j = 0; j < contour_unit_num; ++j) {
                DcmItem* coutour_unit = contour_unit_sequence->getItem(j);
                OFString points_array;
                status = coutour_unit->findAndGetOFStringArray(DCM_ContourData,
                         points_array);

                if (status.bad()) {
                    MI_LOG(MI_ERROR) << "get contour data failed.";
                    return IO_DATA_DAMAGE;
                }

                std::vector<std::string> points;
                boost::split(points, points_array, boost::is_any_of("|/\\"));

                if (0 != points.size() % 3) {
                    MI_LOG(MI_ERROR) << "contour point size invalid.";
                    return IO_DATA_DAMAGE;
                }

                ContourData* contour_data = new ContourData();
                contour_data->points.resize(points.size() / 3);

                for (size_t k = 0; k < points.size() / 3; ++k) {
                    contour_data->points[k]._m[0] = (float)atof(points[k * 3].c_str());
                    contour_data->points[k]._m[1] =
                        (float)atof(points[k * 3 + 1].c_str());
                    contour_data->points[k]._m[2] =
                        (float)atof(points[k * 3 + 2].c_str());
                }

                rt_struct->add_contour(roi_name.c_str(), contour_data);
            }

            MI_LOG(MI_INFO) << "ROI name: " << roi_name;
        }

        return IO_SUCCESS;

    } catch (const Exception& e) {
        rt_struct.reset();
        throw e;
        return IO_FILE_OPEN_FAILED;
    }
}

MED_IMG_END_NAMESPACE