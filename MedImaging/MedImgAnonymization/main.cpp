#include <iostream>
#include <fstream>
#include <sstream>

#include "MedImgIO/mi_dicom_exporter.h"


int main(int argc , char* argv[])
{
    if (argc == 1)
    {
        std::cout << "DICOM Anonymization : "
        std::cout << "arguments[1] be config file.\n";
        return -1;
    }

    if (argc != 4)
    {
        std::cout << "Input valid : \n";
        std::cout << "arguments[1] be src root.\n";
        std::cout << "arguments[2] be dst root.\n";
        std::cout << "arguments[3] be config data.\n";
        return -1;
    }

    const std::string config_file(argv[3]);
    std::cout << "config file : " << config_file << std::endl;
    std::cout << "parsing config file ...\n";
    std::ifstream in(config_file.c_str() , std::ios::in);
    if (!in.is_open())
    {
        std::cout << "open config file : " <<config_file << " failed !\n";
        return -1;
    }

    std::string src_root(argv[1]);
    std::string dst_root(argv[2]);
    std::vector<DcmTagKey> ignore_tags;
    std::string line;
    while(std::getline(in , line))
    {
        std::string tag;
        std::string equal;
        std::string context;
        std::stringstream ss(line);
        ss >> tag >> equal >> context;
        if (tag == "src_root")
        {
            src_root = context;
        }
        else if (tag == "dst_root")
        {
            dst_root = context;
        }
        else if (tag == "skip_derived")
        {
            if (context == "true")
            {

            }
            else
            {

            }
        }
        else if (tag == "patient_name")
        {
            ignore_tags.push_back(DCM_PatientName);
        }
        else if (tag == "patient_id")
        {
            ignore_tags.push_back(DCM_PatientID);
        }
        else if (tag == "patient_birth_date")
        {
            ignore_tags.push_back(DCM_PatientBirthDate);
        }
        else if (tag == "patient_birth_time")
        {
            ignore_tags.push_back(DCM_PatientBirthTime);
        }
        else if (tag == "patient_birth_name")
        {
            ignore_tags.push_back(DCM_PatientBirthName);
        }
        else if (tag == "patient_sex")
        {
            ignore_tags.push_back(DCM_PatientSex);
        }
        else if (tag == "patient_age")
        {
            ignore_tags.push_back(DCM_PatientAge);
        }
        else if (tag == "patient_weight")
        {
            ignore_tags.push_back(DCM_PatientWeight);
        }
        else if (tag == "patient_address")
        {
            ignore_tags.push_back(DCM_PatientAddress);
        }
        else if (tag == "other_patient_ids")
        {
            ignore_tags.push_back(DCM_OtherPatientIDs);
        }
        else if (tag == "other_patient_names")
        {
            ignore_tags.push_back(DCM_OtherPatientNames);
        }
    }

    std::cout << "src root : " << src_root << std::endl;
    std::cout << "dst root : " << dst_root << std::endl;
    std::cout << "DICOM anonymization begin : \n";

    

    std::cout << "DICOM anonymization done : \n";
}