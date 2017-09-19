#include "mi_nodule_anno_config.h"

#include <fstream>

#include "util/mi_configuration.h"
#include "util/mi_string_number_converter.h"

using namespace medical_imaging;

boost::mutex NoduleAnnoConfig::_s_mutex;

NoduleAnnoConfig* NoduleAnnoConfig::_s_instance = nullptr;

NoduleAnnoConfig* NoduleAnnoConfig::instance() {
    if (!_s_instance) {
        boost::unique_lock<boost::mutex> locker(_s_mutex);

        if (!_s_instance) {
            _s_instance = new NoduleAnnoConfig();
        }
    }

    return _s_instance;
}

NoduleAnnoConfig::~NoduleAnnoConfig() {

}


void NoduleAnnoConfig::bind_config_file(const std::string& file) {
    _config_file = file;
}


void NoduleAnnoConfig::initialize()
{
    if (_config_file.empty()) {
        return;
    }

    std::ifstream input_file(_config_file.c_str() , std::ios::in);    
    if (input_file.is_open()) {
        std::string line;
        std::string tag;
        std::string equal;
        std::string context;
        StrNumConverter<float> converter;
        while(std::getline(input_file,line)) {
            std::stringstream ss(line);
            ss >> tag >> equal >> context;
            if (tag == std::string("ProcessingUnit")) {
                if (context == "GPU") {
                    Configuration::instance()->set_processing_unit_type(GPU);
                } else {
                    Configuration::instance()->set_processing_unit_type(CPU);
                }
            } else if (tag == "NoduleOutput") {
                if(context == "TEXT") {
                    this->set_nodule_file_rsa(false);
                } else {
                    this->set_nodule_file_rsa(true);
                }
            } else if (tag == "LastOpenDirection") {
                _last_open_direction = context;
            } else if (tag == "PresetCTAbdomenWW") {
                _preset_windowing[CT_ABDOMEN].first = converter.to_num(context);
            } else if (tag == "PresetCTAbdomenWL") {
                _preset_windowing[CT_ABDOMEN].second = converter.to_num(context);
            } else if (tag == "PresetCTLungsWW") {
                _preset_windowing[CT_LUNGS].first = converter.to_num(context);
            } else if (tag == "PresetCTLungsWL") {
                _preset_windowing[CT_LUNGS].second = converter.to_num(context);
            } else if (tag == "PresetCTBrainWW") {
                _preset_windowing[CT_BRAIN].first = converter.to_num(context);
            } else if (tag == "PresetCTBrainWL") {
                _preset_windowing[CT_BRAIN].second = converter.to_num(context);
            } else if (tag == "PresetCTAngioWW") {
                _preset_windowing[CT_ANGIO].first = converter.to_num(context);
            } else if (tag == "PresetCTAngioWL") {
                _preset_windowing[CT_ANGIO].second = converter.to_num(context);
            } else if (tag == "PresetCTBoneWW") {
                _preset_windowing[CT_BONE].first = converter.to_num(context);
            } else if (tag == "PresetCTBoneWL") {
                _preset_windowing[CT_BONE].second = converter.to_num(context);
            } else if (tag == "PresetCTChestWW") {
                _preset_windowing[CT_CHEST].first = converter.to_num(context);
            } else if (tag == "PresetCTChestWL") {
                _preset_windowing[CT_CHEST].second = converter.to_num(context);
            }  
            else if (tag == "DoubleClickInterval") {
                _double_click_interval = converter.to_num(context);
            } 
        }
        input_file.close();
    }
}

void NoduleAnnoConfig::finalize()
{
    if (_config_file.empty()) {
        return;
    }

    std::ofstream output_file(_config_file.c_str() , std::ios::out);    
    if (output_file.is_open()) {

        output_file << "ProcessingUnit = ";
        if (Configuration::instance()->get_processing_unit_type() == CPU) {
            output_file << "CPU\n";
        } else {
            output_file << "GPU\n";
        }

        if (this->get_nodule_file_rsa()) {
            //output_file << "NoduleOutput = ";
            //output_file << "RSA\n";
        } else {
            output_file << "NoduleOutput = ";
            output_file << "TEXT\n";
        }

        output_file<< "LastOpenDirection = " << _last_open_direction << std::endl; 

        output_file << "PresetCTAbdomenWW = " << _preset_windowing[CT_ABDOMEN].first << std::endl;
        output_file << "PresetCTAbdomenWL = " << _preset_windowing[CT_ABDOMEN].second << std::endl;
        output_file << "PresetCTLungsWW = " << _preset_windowing[CT_LUNGS].first << std::endl;
        output_file << "PresetCTLungsWL = " << _preset_windowing[CT_LUNGS].second << std::endl;
        output_file << "PresetCTBrainWW = " << _preset_windowing[CT_BRAIN].first << std::endl;
        output_file << "PresetCTBrainWL = " << _preset_windowing[CT_BRAIN].second << std::endl;
        output_file << "PresetCTAngioWW = " << _preset_windowing[CT_ANGIO].first << std::endl;
        output_file << "PresetCTAngioWL = " << _preset_windowing[CT_ANGIO].second << std::endl;
        output_file << "PresetCTBoneWW = " << _preset_windowing[CT_BONE].first << std::endl;
        output_file << "PresetCTBoneWL = " << _preset_windowing[CT_BONE].second << std::endl;
        output_file << "PresetCTChestWW = " << _preset_windowing[CT_CHEST].first << std::endl;
        output_file << "PresetCTChestWL = " << _preset_windowing[CT_CHEST].second << std::endl;

        output_file << "DoubleClickInterval = " << _double_click_interval << std::endl;

        output_file.close();
    }
}

NoduleAnnoConfig::NoduleAnnoConfig() {
    init_i();
}

void NoduleAnnoConfig::init_i() {
    _config_file.clear();

    Configuration::instance()->set_processing_unit_type(CPU);

    _is_nodule_file_rsa = true;

    const float PRESET_CT_ABDOMEN_WW = 400;
    const float PRESET_CT_ABDOMEN_WL = 60;
    const float PRESET_CT_LUNGS_WW = 1500;
    const float PRESET_CT_LUNGS_WL = -400;
    const float PRESET_CT_BRAIN_WW = 80;
    const float PRESET_CT_BRAIN_WL = 40;
    const float PRESET_CT_ANGIO_WW = 600;
    const float PRESET_CT_ANGIO_WL = 300;
    const float PRESET_CT_BONE_WW = 1500;
    const float PRESET_CT_BONE_WL = 300;
    const float PRESET_CT_CHEST_WW = 400;
    const float PRESET_CT_CHEST_WL = 40;
    _preset_windowing[CT_ABDOMEN] = std::make_pair(PRESET_CT_ABDOMEN_WW , PRESET_CT_ABDOMEN_WL);
    _preset_windowing[CT_LUNGS] = std::make_pair(PRESET_CT_LUNGS_WW , PRESET_CT_LUNGS_WL);
    _preset_windowing[CT_BRAIN] = std::make_pair(PRESET_CT_BRAIN_WW , PRESET_CT_BRAIN_WL);
    _preset_windowing[CT_ANGIO] = std::make_pair(PRESET_CT_ANGIO_WW , PRESET_CT_ANGIO_WL);
    _preset_windowing[CT_BONE] = std::make_pair(PRESET_CT_BONE_WW , PRESET_CT_BONE_WL);
    _preset_windowing[CT_CHEST] = std::make_pair(PRESET_CT_CHEST_WW , PRESET_CT_CHEST_WL);

    _last_open_direction.clear();

    _double_click_interval = 0;
}

void NoduleAnnoConfig::set_nodule_file_rsa(bool b) {
    _is_nodule_file_rsa = b;
}

bool NoduleAnnoConfig::get_nodule_file_rsa() {
    return _is_nodule_file_rsa;
}

void NoduleAnnoConfig::set_preset_wl(PreSetWLType type, float ww, float wl) {
    _preset_windowing[type].first = ww;
    _preset_windowing[type].second = wl;
}

void NoduleAnnoConfig::get_preset_wl(PreSetWLType type, float& ww, float& wl) const {
    const std::map<PreSetWLType , std::pair<float, float>>::const_iterator it = _preset_windowing.find(type);
    ww = it->second.first;
    wl = it->second.second;
}

void NoduleAnnoConfig::set_last_open_direction(const std::string& path)
{
    _last_open_direction = path;
}

std::string NoduleAnnoConfig::get_last_open_direction() const
{
    return _last_open_direction;
}

int NoduleAnnoConfig::get_double_click_interval() const
{
    return _double_click_interval;
}