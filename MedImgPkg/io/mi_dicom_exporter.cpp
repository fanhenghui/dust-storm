#include "mi_dicom_exporter.h"

#include "dcmtk/config/osconfig.h"
#include "dcmtk/oflog/oflog.h"
#include "dcmtk/ofstd/ofconsol.h" /* for ofConsole */

#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmdata/dcdeftag.h" /* for tag constants */
#include "dcmtk/dcmdata/dcdicdir.h"
#include "dcmtk/dcmdata/dcistrmf.h"
#include "dcmtk/dcmdata/dcitem.h"  /* for DcmItem */
#include "dcmtk/dcmdata/dcpixel.h" /* for DcmPixelData */
#include "dcmtk/dcmdata/dcpxitem.h"
#include "dcmtk/dcmdata/dcsequen.h" /* for DcmSequenceOfItems */
#include "dcmtk/dcmdata/dcswap.h"   /* for swapIfNecessary() */
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmdata/dcuid.h" /* for dcmGenerateUniqueIdentifier()*/

#include "dcmtk/dcmdata/dcrledrg.h"  // Support for RLE images
#include "dcmtk/dcmimage/diqtcmap.h" /* for DcmQuantColorMapping */
#include "dcmtk/dcmimage/diqtctab.h" /* for DcmQuantColorTable */
#include "dcmtk/dcmimage/diqtfs.h"   /* for DcmQuantFloydSteinberg */
#include "dcmtk/dcmimage/diqthash.h" /* for DcmQuantColorHashTable */
#include "dcmtk/dcmimage/diqtid.h"   /* for DcmQuantIdent */
#include "dcmtk/dcmimage/diqtpix.h"  /* for DcmQuantPixel */
#include "dcmtk/dcmimage/diregist.h" // Support for color images
#include "dcmtk/dcmimgle/dcmimage.h" /* for DicomImage */

#include "dcmtk/dcmjpeg/djdecode.h" // Support for JPEG images
#include "dcmtk/dcmjpeg/djrplol.h"

#include <iostream>
#include <stdio.h>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

#include "util/mi_model_progress.h"

MED_IMG_BEGIN_NAMESPACE

DICOMExporter::DICOMExporter() : _progress(0.0f), _skip_derived_image(false) {
    _taglist.clear();
    _taglist.push_back(DCM_PatientName);
    _taglist.push_back(DCM_PatientID);
    _taglist.push_back(DCM_PatientBirthDate);
    _taglist.push_back(DCM_PatientBirthTime);
    _taglist.push_back(DCM_PatientBirthName);
    _taglist.push_back(DCM_PatientSex);
    _taglist.push_back(DCM_PatientAge);
    _taglist.push_back(DCM_PatientWeight);
    _taglist.push_back(DCM_PatientAddress);
    //_taglist.push_back(DCM_OtherPatientIDs);
    _taglist.push_back(DCM_OtherPatientNames);
}

DICOMExporter::~DICOMExporter() {}

void DICOMExporter::set_progress_model(std::shared_ptr<ProgressModel> model) {
    _model = model;
}

void DICOMExporter::set_anonymous_taglist(
    const std::vector<DcmTagKey>& taglist) {
    _taglist.clear();
    _taglist = taglist;
}

IOStatus DICOMExporter::load_dicom_file(const std::string file_name,
                                        DcmFileFormatPtr& fileformatptr) {
    fileformatptr.reset(new DcmFileFormat());
    DcmDataset* data_set = nullptr;

    if (fileformatptr->loadFile(file_name.c_str()).good()) {
        // Check compressed
        DcmMetaInfo* meta_info = fileformatptr->getMetaInfo();

        if (!meta_info) {
            return IO_DATA_DAMAGE;
        }

        OFString context;
        OFCondition status =
            meta_info->findAndGetOFString(DCM_TransferSyntaxUID, context);

        if (status.bad()) {
            return IO_DATA_DAMAGE;
        }

        const std::string my_tsu(context.c_str());

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
        // Lossless JPEG Image
        // Compression
        const std::string TSU_JPEGProcess14TransferSyntax =
            std::string("1.2.840.10008.1.2.4.57");

        // JEPG2000 dcmtk
        const std::string TSU_JEPG2000CompressionLosslessOnly =
            std::string("1.2.840.10008.1.2.4.90");
        const std::string TSU_JEPG2000Compression =
            std::string("1.2.840.10008.1.2.4.91");

        if (my_tsu == TSU_LittleEndianImplicitTransferSyntax ||
                my_tsu == TSU_LittleEndianExplicitTransferSyntax ||
                my_tsu == TSU_DeflatedExplicitVRLittleEndianTransferSyntax ||
                my_tsu == TSU_BigEndianExplicitTransferSyntax) {
            return IO_SUCCESS;
        } else if (my_tsu == TSU_JPEGProcess14SV1TransferSyntax ||
                   my_tsu == TSU_JPEGProcess14TransferSyntax) {
            // check if the input DICOM is compressed
            DJDecoderRegistration::registerCodecs(); // register JPEG codecs
            data_set = fileformatptr->getDataset();
            // decompress data set if compressed
            data_set->chooseRepresentation(EXS_LittleEndianExplicit, NULL);

            // check if everything went well
            if (data_set->canWriteXfer(EXS_LittleEndianExplicit)) {
                fileformatptr->saveFile("test_decompressed.dcm",
                                        EXS_LittleEndianExplicit);
            }

            DJDecoderRegistration::cleanup(); // deregister JPEG codecs

            fileformatptr.reset(new DcmFileFormat());
            fileformatptr->loadFile("test_decompressed.dcm");

            return IO_SUCCESS;
        } else if (my_tsu == TSU_JEPG2000CompressionLosslessOnly ||
                   my_tsu == TSU_JEPG2000Compression) {
            return IO_UNSUPPORTED_YET;
        } else {
            return IO_UNSUPPORTED_YET;
        }
    } else {
        return IO_DATA_DAMAGE;
    }
}

IOStatus MED_IMG_NAMESPACE::DICOMExporter::export_series(
    const std::vector<std::string>& in_files,
    const std::vector<std::string>& out_files, ExportDicomDataType etype) {
    if (in_files.empty() || out_files.empty() ||
            in_files.size() != out_files.size()) {
        set_progress_i(100);
        return IO_EMPTY_INPUT;
    }

    DcmFileFormatPtr fileformat_ptr;
    std::string file_name;
    DcmDataset* data_set = nullptr;
    int progress_step =
        static_cast<int>(std::ceil(double(in_files.size()) / 10.0));
    int progress = 0;

    set_progress_i(0);

    IOStatus status;

    switch (etype) {
    case EXPORT_ORIGINAL_DICOM:
        for (int i = 0; i < in_files.size(); ++i) {
            status = load_dicom_file(in_files[i], fileformat_ptr);

            if (status != IO_SUCCESS) {
                return status;
            }

            fileformat_ptr->saveFile(out_files[i].c_str(), EXS_LittleEndianExplicit);

            if (i % progress_step == 1) {
                set_progress_i(progress);
                progress += 10;
            }
        }

        break;

    case EXPORT_ANONYMOUS_DICOM:
        for (int i = 0; i < in_files.size(); ++i) {
            status = load_dicom_file(in_files[i], fileformat_ptr);

            if (status != IO_SUCCESS) {
                return status;
            }

            // Check image type
            OFString context;

            if (fileformat_ptr->getDataset()
                    ->findAndGetOFStringArray(DCM_ImageType, context)
                    .good()) {
                std::string image_type(context.c_str());

                if (_skip_derived_image &&
                        image_type.find("SECONDARY") !=
                        image_type.npos) { // Skip derived image
                    continue;
                }
            } else {
                return IO_UNSUPPORTED_YET;
            }

            anonymous_dicom_data(fileformat_ptr);
            fileformat_ptr->saveFile(out_files[i].c_str(), EXS_LittleEndianExplicit);

            if (i % progress_step == 1) {
                set_progress_i(progress);
                progress += 10;
            }
        }

        break;

    case EXPORT_ANONYMOUS_DICOM_WITHOUT_PRIVATETAG:
        for (int i = 0; i < in_files.size(); ++i) {
            status = load_dicom_file(in_files[i], fileformat_ptr);

            if (status != IO_SUCCESS) {
                return status;
            }

            // Check image type
            OFString context;

            if (fileformat_ptr->getDataset()
                    ->findAndGetOFStringArray(DCM_ImageType, context)
                    .good()) {
                std::string image_type(context.c_str());

                if (_skip_derived_image &&
                        image_type.find("SECONDARY") !=
                        image_type.npos) { // Skip derived image
                    continue;
                }
            } else {
                return IO_UNSUPPORTED_YET;
            }

            anonymous_dicom_data(fileformat_ptr);
            remove_private_tag(fileformat_ptr);
            fileformat_ptr->saveFile(out_files[i].c_str(), EXS_LittleEndianExplicit);

            if (i % progress_step == 1) {
                set_progress_i(progress);
                progress += 10;
            }
        }

        break;

    case EXPORT_RAW:
        break;

    case EXPORT_BITMAP:
        for (int i = 0; i < in_files.size(); ++i) {
            status = save_dicom_as_bitmap(in_files[i], out_files[i]);

            if (status != IO_SUCCESS) {
                return status;
            }

            if (i % progress_step == 1) {
                set_progress_i(progress);
                progress += 10;
            }
        }

        break;

    default:
        break;
    }

    set_progress_i(100);

    return IO_SUCCESS;
}

IOStatus DICOMExporter::save_dicom_as_bitmap(const std::string in_file_name,
        const std::string out_file_name) {
    DcmFileFormatPtr fileformat_ptr;
    IOStatus status = load_dicom_file(in_file_name, fileformat_ptr);

    if (status != IO_SUCCESS) {
        return status;
    }

    DcmDataset* data_set = fileformat_ptr->getDataset();

    // Save BITMAP
    DicomImage dcm_image(data_set, EXS_LittleEndianExplicit);
    dcm_image.writeBMP(out_file_name.c_str());
    data_set = nullptr;
    delete data_set;

    return IO_SUCCESS;
}

void DICOMExporter::anonymous_dicom_data(DcmFileFormatPtr in_fileformat_ptr) {
    for (int i = 0; i < _taglist.size(); ++i) {
        in_fileformat_ptr->getDataset()->putAndInsertString(_taglist[i], " ");
    }

    anonymous_all_patient_name(in_fileformat_ptr);
}

void DICOMExporter::anonymous_all_patient_name(
    DcmFileFormatPtr in_fileformat_ptr) {
    DcmItem dcm_item = *in_fileformat_ptr->getDataset();
    DcmStack stack;
    DcmObject* dcm_obj = NULL;
    DcmTagKey tag;
    OFCondition status = dcm_item.nextObject(stack, OFTrue);

    while (status.good()) {
        dcm_obj = stack.top();
        tag = dcm_obj->getTag();

        if (tag.getGroup() &
                1) { // private tag ? // all private data has an odd group number
            stack.pop();
            delete((DcmItem*)(stack.top()))->remove(dcm_obj);
        }

        DcmVR dcm_vr = dcm_obj->getVR();
        OFString dcm_vr_name = dcm_vr.getVRName();

        if (dcm_vr_name == "PN") {
            //            delete ((DcmItem *)(stack.top()))->remove(dobj);
            in_fileformat_ptr->getDataset()->putAndInsertString(tag, "Anonymous");
            //            cout << VRName << endl;
        }

        status = dcm_item.nextObject(stack, OFTrue);
    }
}

void DICOMExporter::remove_private_tag(DcmFileFormatPtr in_fileformat_ptr) {
    DcmItem dcm_item = *in_fileformat_ptr->getDataset();
    DcmStack stack;
    DcmObject* dcm_obj = NULL;
    DcmTagKey tag;
    OFCondition status = dcm_item.nextObject(stack, OFTrue);

    while (status.good()) {
        dcm_obj = stack.top();
        tag = dcm_obj->getTag();

        if (tag.getGroup() &
                1) { // private tag ? // all private data has an odd group number
            stack.pop();
            delete((DcmItem*)(stack.top()))->remove(dcm_obj);
        }

        status = dcm_item.nextObject(stack, OFTrue);
    }

    dcm_obj = nullptr;
    delete dcm_obj;
}

IOStatus DICOMExporter::save_dicom_as_raw(const std::string in_file_name,
        const std::string out_file_name) {
    return IO_SUCCESS;
}

void DICOMExporter::add_progress_i(float value) {
    if (_model) {
        _progress += value;
        int progress = static_cast<int>(_progress);
        progress = progress > 100 ? 100 : progress;
        _model->set_progress(progress);
        _model->notify();
    }
}

void DICOMExporter::set_progress_i(int value) {
    if (_model) {
        _progress = static_cast<float>(value);
        _model->set_progress(value);
        _model->notify();
    }
}

void DICOMExporter::skip_derived_image(bool flag) {
    _skip_derived_image = flag;
}

MED_IMG_END_NAMESPACE
