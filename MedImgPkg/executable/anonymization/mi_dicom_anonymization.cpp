#include "boost/filesystem.hpp"
#include <fstream>
#ifdef WIN32
#include <io.h>
#endif
#include <iostream>
#include <sstream>

#include "io/mi_dicom_exporter.h"

static std::ofstream out_log;

class LogSheild {
public:
    LogSheild() {
        out_log.open("anon.log", std::ios::out);

        if (out_log.is_open()) {
            out_log << "DICOM anonymization log:\n";
        }
    }
    ~LogSheild() {
        out_log.close();
    }

protected:
private:
};

void get_all_files(const std::string& root, unsigned int& num) {
    if (root.empty()) {
        return;
    } else {
        std::vector<std::string> dirs;

        for (boost::filesystem::directory_iterator it(root);
                it != boost::filesystem::directory_iterator(); ++it) {
            if (boost::filesystem::is_directory(*it)) {
                dirs.push_back(it->path().filename().string());
            } else {
                const std::string ext = boost::filesystem::extension(*it);

                if (ext == ".dcm" || ext == ".DCM" || ext == ".Dcm") {
                    ++num;
                }
            }
        }

        for (unsigned int i = 0; i < dirs.size(); ++i) {
            const std::string next_dir(root + "/" + dirs[i]);

            get_all_files(next_dir, num);
        }
    }
}

int anon(const std::string& src_root, const std::string& dst_root,
         const std::vector<DcmTagKey>& ignore_tags, bool skip_derived,
         int& file_id, int file_sum) {
    if (src_root.empty()) {
        return 0;
    }

    std::vector<std::string> dirs;
    std::vector<std::string> files;

    for (boost::filesystem::directory_iterator it(src_root);
            it != boost::filesystem::directory_iterator(); ++it) {
        if (boost::filesystem::is_directory(*it)) {
            dirs.push_back(it->path().filename().string());
        } else {
            const std::string ext = boost::filesystem::extension(*it);

            if (ext == ".dcm" || ext == ".DCM" || ext == ".Dcm") {
                files.push_back(it->path().filename().string());
            }
        }
    }

    for (unsigned int i = 0; i < dirs.size(); ++i) {
        const std::string src_dir(src_root + "/" + dirs[i]);
        const std::string dst_dir(dst_root + "/" + dirs[i]);

        boost::filesystem::path path(dst_dir);
        boost::filesystem::create_directories(path);

        int status =
            anon(src_dir, dst_dir, ignore_tags, skip_derived, file_id, file_sum);

        if (0 != status) {
            return status;
        }
    }

    for (unsigned int i = 0; i < files.size(); ++i) {
        const std::string src_file(src_root + "/" + files[i]);
        const std::string dst_file(dst_root + "/" + files[i]);

        medical_imaging::DICOMExporter exporter;
        exporter.set_anonymous_taglist(ignore_tags);
        exporter.skip_derived_image(skip_derived);
        medical_imaging::IOStatus status =
            exporter.export_series(std::vector<std::string>(1, src_file),
                                   std::vector<std::string>(1, dst_file),
                                   medical_imaging::EXPORT_ANONYMOUS_DICOM);

        if (status != medical_imaging::IO_SUCCESS) {
            if (status == medical_imaging::IO_DATA_DAMAGE) {
                out_log << "Data damage : " << src_file << std::endl;
            } else if (status == medical_imaging::IO_UNSUPPORTED_YET) {
                out_log << "Unsupported DICOM format : " << src_file << std::endl;
            } else {
                out_log << "Other error : " << src_file << std::endl;
            }
        }

        ++file_id;
        int sub = file_sum / 20;

        if (sub != 0) {
            int k = file_id / sub;

            if (k != 0 && k * sub == file_id) {
                std::cout << ">";
            }
        }
    }

    return 0;
}

void create_empty_config_file() {
    const std::string config_file = "config.txt";
    std::ofstream out(config_file.c_str(), std::ios::out);

    if (out.is_open()) {
        out << "src_root = E:/Data/MyData\n\
dst_root = E:/Data/MyDataAnon\n\
skip_derived = true\n\
remove_patient_name = true\n\
remove_patient_id = true\n\
remove_patient_birth_date = true\n\
remove_patient_birth_time = true\n\
remove_patient_birth_name = true\n\
remove_patient_sex = true\n\
remove_patient_age = true\n\
remove_patient_weight = true\n\
remove_patient_address = true\n\
remove_other_patient_ids = true\n\
remove_other_patient_names = true\n\
remove_manufacturer = true\n\
remove_manufacturer_model_name = true\n\
remove_institution_name = true\n\
remove_institution_address = true\n\
remove_institutional_department_name = true\n\
remove_referring_physician_name = true\n\
remove_operator_name = true\n\
remove_station_name = true";
        out.close();
    }
}

int main(int argc, char* argv[]) {

    LogSheild log_sheild;

    std::string config_file;

    if (argc == 1) {
        std::cout << "DICOM Anonymization : ";
        std::cout << "arguments[1] should be config file.\n";
        std::cout << "Take default config file name config.txt.\n";

        out_log << "arguments[1] should be config file.\n";
        out_log << "Take default config file name config.txt.\n";

        config_file = "config.txt";
    } else if (argc == 2) {
        config_file = std::string(argv[1]);
    }

    std::cout << "config file : " << config_file << std::endl;
    std::cout << "parsing config file ...\n";

    out_log << "config file : " << config_file << std::endl;

    std::ifstream in(config_file.c_str(), std::ios::in);

    if (!in.is_open()) {
        std::cout << "open config file : " << config_file << " failed !\n";
        out_log << "open config file : " << config_file << " failed !\n";

        if (argc == 1) {
            out_log << "Create a empty config file for you.";
            create_empty_config_file();
        }

        return -1;
    }

    std::string src_root;
    std::string dst_root;
    bool skip_derived = false;
    std::vector<DcmTagKey> ignore_tags;
    std::string line;

    while (std::getline(in, line)) {
        std::string tag;
        std::string equal;
        std::string context;
        std::stringstream ss(line);
        ss >> tag >> equal >> context;

        if (tag == "src_root") {
            src_root = context;
        } else if (tag == "dst_root") {
            dst_root = context;
        } else if (tag == "skip_derived") {
            if (context == "true") {
                skip_derived = true;
            } else {
                skip_derived = false;
            }
        } else if (tag == "remove_patient_name" && context == "true") {
            ignore_tags.push_back(DCM_PatientName);
        } else if (tag == "remove_patient_id" && context == "true") {
            ignore_tags.push_back(DCM_PatientID);
        } else if (tag == "remove_patient_birth_date" && context == "true") {
            ignore_tags.push_back(DCM_PatientBirthDate);
        } else if (tag == "remove_patient_birth_time" && context == "true") {
            ignore_tags.push_back(DCM_PatientBirthTime);
        } else if (tag == "remove_patient_birth_name" && context == "true") {
            ignore_tags.push_back(DCM_PatientBirthName);
        } else if (tag == "remove_patient_sex" && context == "true") {
            ignore_tags.push_back(DCM_PatientSex);
        } else if (tag == "remove_patient_age" && context == "true") {
            ignore_tags.push_back(DCM_PatientAge);
        } else if (tag == "remove_patient_weight" && context == "true") {
            ignore_tags.push_back(DCM_PatientWeight);
        } else if (tag == "remove_patient_address" && context == "true") {
            ignore_tags.push_back(DCM_PatientAddress);
        } else if (tag == "remove_other_patient_ids" && context == "true") {
            ignore_tags.push_back(DCM_OtherPatientIDs);
        } else if (tag == "remove_other_patient_names" && context == "true") {
            ignore_tags.push_back(DCM_OtherPatientNames);
        } else if (tag == "remove_manufacturer" && context == "true") {
            ignore_tags.push_back(DCM_Manufacturer);
        } else if (tag == "remove_manufacturer_model_name" && context == "true") {
            ignore_tags.push_back(DCM_ManufacturerModelName);
        } else if (tag == "remove_institution_name" && context == "true") {
            ignore_tags.push_back(DCM_InstitutionName);
        } else if (tag == "remove_institution_address" && context == "true") {
            ignore_tags.push_back(DCM_InstitutionAddress);
        } else if (tag == "remove_institutional_department_name" &&
                   context == "true") {
            ignore_tags.push_back(DCM_InstitutionalDepartmentName);
        } else if (tag == "remove_referring_physician_name" && context == "true") {
            ignore_tags.push_back(DCM_ReferringPhysicianName);
        } else if (tag == "remove_operator_name" && context == "true") {
            ignore_tags.push_back(DCM_OperatorsName);
        } else if (tag == "remove_station_name" && context == "true") {
            ignore_tags.push_back(DCM_StationName);
        }
    }

    std::cout << "src root : " << src_root << std::endl;
    std::cout << "dst root : " << dst_root << std::endl;
    std::cout << "DICOM anonymization begin : \n";

    out_log << "src root : " << src_root << std::endl;
    out_log << "dst root : " << dst_root << std::endl;
    out_log << "DICOM anonymization begin : \n";

    if (src_root.empty()) {
        out_log << "src root is empty!\n";
        std::cout << "src root is empty!\n";
        return -1;
    }

    if (dst_root.empty()) {
        out_log << "dst root is empty!\n";
        std::cout << "dst root is empty!\n";
        return -1;
    }

    unsigned int file_sum = 0;
    get_all_files(src_root, file_sum);
    std::cout << "File number is : " << file_sum << std::endl;
    out_log << "File number is : " << file_sum << std::endl;

    if (0 == file_sum) {
        out_log << "Empty file collection.\n";
        return -1;
    }

    int cur_file_id(0);
    int status =
        anon(src_root, dst_root, ignore_tags, skip_derived, cur_file_id, file_sum);

    std::cout << std::endl;

    if (status != 0 || cur_file_id != (int)file_sum) {
        out_log << "DICOM anonymization failed : \n";
        std::cout << "DICOM anonymization failed : \n";
        return -1;
    } else {
        out_log << "DICOM anonymization done : \n";
        std::cout << "DICOM anonymization done : \n";
        return 0;
    }
}