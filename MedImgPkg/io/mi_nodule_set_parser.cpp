#include "mi_nodule_set_parser.h"

#include <set>

#include "boost/algorithm/string.hpp"
#include "boost/format.hpp"
#include "boost/tokenizer.hpp"

#include "arithmetic/mi_rsa_utils.h"
#include "util/mi_string_number_converter.h"

#include "mi_nodule_set.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
struct NoduleUnit {
    int nodule_id;                // none encoded
    unsigned char series_id[512]; // none encoded
    unsigned char pos_x[512];
    unsigned char pos_y[512];
    unsigned char pos_z[512];
    unsigned char diameter[512];
    unsigned char type[512];

    NoduleUnit() {
        nodule_id = 0;
        memset(series_id, 0, sizeof(series_id));
        memset(pos_x, 0, sizeof(pos_x));
        memset(pos_y, 0, sizeof(pos_y));
        memset(pos_z, 0, sizeof(pos_z));
        memset(diameter, 0, sizeof(diameter));
        memset(type, 0, sizeof(type));
    }
};
}

NoduleSetParser::NoduleSetParser() {}

NoduleSetParser::~NoduleSetParser() {}

IOStatus NoduleSetParser::load_as_csv(const std::string& file_path,
                                      std::shared_ptr<NoduleSet>& nodule_set) {
    ////////////////////////////////////////////////////////////////////////
    //Columen char
    //seriesuid: (Not NULL) series ID
    //noduleuid: (optional) useless yet
    //coordX: (Not NULL) sphere center X
    //coordY: (Not NULL) sphere center Y
    //coordZ: (Not NULL) sphere center Z
    //diameter_mm/radius: (Alternative) LUNA lung set use diameter_mm; AI evaluation use radius
    //Type: (optinal) user annotation set; AI can't evaluate yet)
    //probability:(optinal) AI evaluated
    ////////////////////////////////////////////////////////////////////////
    typedef struct InnerItem_ {
        std::string series_id;
        double cx;
        double cy;
        double cz;
        double diameter;
        double probability;
        std::string type;
    
        InnerItem_():cx(0),cy(0),cz(0),diameter(0),probability(1.0) {};
    } InnerItem;
    std::vector<InnerItem> items;
    std::vector<std::string> column_name;
    std::set<std::string> column_set;

    std::ifstream in(file_path.c_str(), std::ios::in);
    if (!in.is_open()) {
        return IO_FILE_OPEN_FAILED;
    }
    // series uid,nodule uid,coordX,coordY,coordZ,diameter_mm,Type
    std::string line;
    std::getline(in, line);
    std::vector<std::string> infos;
    boost::split(infos, line, boost::is_any_of(","));
    
    for (size_t i = 0; i < infos.size(); ++i) {
        column_set.insert(infos[i]);
        column_name.push_back(infos[i]);
    }
    const size_t column_num = column_set.size();

    //check header(not null)
    const static int COLUMNS_NOT_NULL_NUM = 4;
    const static std::string COLUMNS_NOT_NULL_STR[COLUMNS_NOT_NULL_NUM] = {"seriesuid", "coordX", "coordY", "coordZ"};  
    int pass = 0;
    for (int i = 0; i<4; ++i ) {
        if (column_set.find(COLUMNS_NOT_NULL_STR[i]) != column_set.end()) {
            ++pass;
        } else {
            break;
        }
    }
    if(pass != COLUMNS_NOT_NULL_NUM) {
        MI_IO_LOG(MI_ERROR) << "csv file header column check failed.";
        in.close();
        return IO_DATA_DAMAGE;
    }
    //check radius or diameter
    if (!(column_set.find("radius") != column_set.end() || 
          column_set.find("diameter_mm") != column_set.end()) ) {
        in.close();
        MI_IO_LOG(MI_ERROR) << "csv file has no radius/diameter_mm column.";
        return IO_DATA_DAMAGE;
    }

    StrNumConverter<double> num_converter;    
    while (std::getline(in, line)) {
        std::vector<std::string> infos;
        boost::split(infos, line, boost::is_any_of(","));

        if (column_num != infos.size()) {
            MI_IO_LOG(MI_ERROR) << "csv file column damage: " << line;
            continue;
        }

        std::string item_name;
        InnerItem item;
        for(size_t i=0; i < infos.size(); ++i) { 
            item_name = column_name[i];
            const std::string& column_value = infos[i];
            if (item_name == "seriesuid") {
                item.series_id = column_value;
            } else if (item_name == "coordX") {
                item.cx = num_converter.to_num(column_value);
            } else if (item_name == "coordY") {
                item.cy = num_converter.to_num(column_value);
            } else if (item_name == "coordZ") {
                item.cz = num_converter.to_num(column_value);
            } else if (item_name == "radius") {
                item.diameter = num_converter.to_num(column_value)*2.0;
            } else if (item_name == "diameter_mm") {
                item.diameter = num_converter.to_num(column_value);
            } else if (item_name == "probability") {
                item.probability = num_converter.to_num(column_value);
            } else if (item_name == "type" || item_name == "Type") {
                item.type = column_value;
            }
        }

        if (_series_id.empty()) {
            items.push_back(item);
        } else if( item.series_id == _series_id) {
            items.push_back(item);
        }
    }

    in.close();

    if (items.empty()) {
        //TODO 这种情况怎么处理(没有数据条目)
        MI_IO_LOG(MI_WARNING) << "load empty csv result.";
        return IO_SUCCESS;
    }

    nodule_set->clear_nodule();
    for (auto it = items.begin(); it != items.end(); ++it) {
        const InnerItem& item = *it;
        VOISphere voi(Point3(item.cx, item.cy, item.cz), item.diameter, item.type);\
        voi.probability = item.probability;
        nodule_set->add_nodule(voi);
    }

    return IO_SUCCESS;
}

IOStatus NoduleSetParser::save_as_csv(const std::string& file_path,
                             const std::shared_ptr<NoduleSet>& nodule_set) {
    IO_CHECK_NULL_EXCEPTION(nodule_set);

    std::fstream out(file_path.c_str(), std::ios::out);

    if (!out.is_open()) {
        return IO_FILE_OPEN_FAILED;
    } else {
        const std::vector<VOISphere>& nodules = nodule_set->get_nodule_set();        
        out << "seriesuid,coordX,coordY,coordZ,radius";
        bool has_type = false;
        bool has_probablity = false;
        float probability_sum = 0;
        for (auto it = nodules.begin(); it != nodules.end(); ++it) {
            probability_sum += (*it).probability;
            if (!has_type && !(*it).name.empty()) {
                has_type = true;
            }
        }
        has_probablity = probability_sum > 0;
        if (has_probablity) {
            out << ",probability";
        }
        if (has_type) {
            out << ",type";
        }
        out << std::endl; 

        
        out << std::fixed;

        for (auto it = nodules.begin(); it != nodules.end(); ++it) {
            const VOISphere& voi = *it;
            out << _series_id << "," << voi.center.x << ","
                << voi.center.y << "," << voi.center.z << "," << voi.diameter*0.5f;
            if (has_probablity)  {
               out << "," << voi.probability;
            }
            if (has_type) {
                out << "," << voi.name;
            }
            out << std::endl;
        }

        out.close();
    }

    return IO_SUCCESS;
}

IOStatus NoduleSetParser::load_as_rsa_binary(const std::string& file_path,
                                    const mbedtls_rsa_context& rsa,
                                    std::shared_ptr<NoduleSet>& nodule_set) {
    IO_CHECK_NULL_EXCEPTION(nodule_set);
    nodule_set->clear_nodule();

    std::fstream in(file_path, std::ios::in | std::ios::binary);

    if (!in.is_open()) {
        return IO_FILE_OPEN_FAILED;
    }

    RSAUtils rsa_utils;
    StrNumConverter<double> str_num_convertor;
    int status(0);

    // 1 Read nodule number
    unsigned char buffer[1024];
    unsigned char input_nudule_num[512];

    if (!in.read((char*)input_nudule_num, sizeof(input_nudule_num))) {
        in.close();
        return IO_DATA_DAMAGE;
    }

    memset(buffer, 0, sizeof(buffer));
    status = rsa_utils.detrypt(rsa, 512, input_nudule_num, buffer);

    if (status != 0) {
        in.close();
        return IO_ENCRYPT_FAILED;
    }

    const int num = static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

    if (num < 0) { // warning no nodule
        in.close();
        return IO_SUCCESS;
    }

    for (int i = 0; i < num; ++i) {
        NoduleUnit nodule_unit;

        if (!in.read((char*)(&nodule_unit), sizeof(nodule_unit))) {
            break;
        }

        // Check series id
        if (!_series_id.empty()) {
            if (nodule_unit.series_id[_series_id.size()] != '\0') {
                return IO_UNMATCHED_FILE;
            }

            size_t j = 0;
            for (; j < 512 && j < _series_id.size(); ++i) {
                if (nodule_unit.series_id[j] == '\0' ||
                        nodule_unit.series_id[j] != _series_id[j]) {
                    break;
                }
            }

            if (j != _series_id.size()) {
                return IO_UNMATCHED_FILE;
            }
        }

        memset(buffer, 0, sizeof(buffer));
        status = rsa_utils.detrypt(rsa, 512, nodule_unit.pos_x, buffer);

        if (status != 0) {
            in.close();
            return IO_ENCRYPT_FAILED;
        }

        double pos_x =
            static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer, 0, sizeof(buffer));
        status = rsa_utils.detrypt(rsa, 512, nodule_unit.pos_y, buffer);

        if (status != 0) {
            in.close();
            return IO_ENCRYPT_FAILED;
        }

        double pos_y =
            static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer, 0, sizeof(buffer));
        status = rsa_utils.detrypt(rsa, 512, nodule_unit.pos_z, buffer);

        if (status != 0) {
            in.close();
            return IO_ENCRYPT_FAILED;
        }

        double pos_z =
            static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer, 0, sizeof(buffer));
        status = rsa_utils.detrypt(rsa, 512, nodule_unit.diameter, buffer);

        if (status != 0) {
            in.close();
            return IO_ENCRYPT_FAILED;
        }

        double diameter =
            static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer, 0, sizeof(buffer));
        status = rsa_utils.detrypt(rsa, 512, nodule_unit.type, buffer);

        if (status != 0) {
            in.close();
            return IO_ENCRYPT_FAILED;
        }

        std::string type = std::string((char*)buffer);

        nodule_set->add_nodule(
            VOISphere(Point3(pos_x, pos_y, pos_z), diameter, type));
    }

    in.close();
    return IO_SUCCESS;
}

IOStatus NoduleSetParser::save_as_rsa_binary(
    const std::string& file_path, const mbedtls_rsa_context& rsa,
    const std::shared_ptr<NoduleSet>& nodule_set) {
    IO_CHECK_NULL_EXCEPTION(nodule_set);

    std::fstream out(file_path, std::ios::out | std::ios::binary);

    if (!out.is_open()) {
        return IO_FILE_OPEN_FAILED;
    }

    RSAUtils rsa_utils;
    StrNumConverter<double> str_num_convertor;
    int status(0);

    // 1 Write nodule number
    unsigned char output_nudule_num[512];
    memset(output_nudule_num, 0, sizeof(output_nudule_num));

    const std::vector<VOISphere>& nodules = nodule_set->get_nodule_set();
    std::string nodule_num =
        str_num_convertor.to_string(static_cast<double>(nodules.size()));

    status = rsa_utils.entrypt(rsa, nodule_num.size(),
                               (unsigned char*)(nodule_num.c_str()),
                               output_nudule_num);

    if (status != 0) {
        out.close();
        return IO_ENCRYPT_FAILED;
    }

    out.write((char*)output_nudule_num, sizeof(output_nudule_num));

    // 2 Save nodule number
    int id = 0;

    for (auto it = nodules.begin(); it != nodules.end(); ++it) {
        const VOISphere& voi = *it;
        std::string pos_x = str_num_convertor.to_string(voi.center.x);
        std::string pos_y = str_num_convertor.to_string(voi.center.y);
        std::string pos_z = str_num_convertor.to_string(voi.center.z);
        std::string diameter = str_num_convertor.to_string(voi.diameter);

        NoduleUnit nodule_unit;

        nodule_unit.nodule_id = id++;

        if (!_series_id.empty()) {
            if (_series_id.size() > 511) {
                return IO_UNSUPPORTED_YET;
            }

            for (size_t i = 0; i < _series_id.size(); ++i) {
                nodule_unit.series_id[i] = _series_id[i];
            }

            nodule_unit.series_id[_series_id.size()] = '\0';
        }

        status = rsa_utils.entrypt(
                     rsa, pos_x.size(), (unsigned char*)(pos_x.c_str()), nodule_unit.pos_x);

        if (status != 0) {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(
                     rsa, pos_y.size(), (unsigned char*)(pos_y.c_str()), nodule_unit.pos_y);

        if (status != 0) {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(
                     rsa, pos_z.size(), (unsigned char*)(pos_z.c_str()), nodule_unit.pos_z);

        if (status != 0) {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(rsa, diameter.size(),
                                   (unsigned char*)(diameter.c_str()),
                                   nodule_unit.diameter);

        if (status != 0) {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(rsa, voi.name.size(),
                                   (unsigned char*)(voi.name.c_str()),
                                   nodule_unit.type);

        if (status != 0) {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        out.write((char*)(&nodule_unit), sizeof(nodule_unit));
    }

    out.close();

    return IO_SUCCESS;
}

void NoduleSetParser::set_series_id(const std::string& series_id) {
    _series_id = series_id;
}

MED_IMG_END_NAMESPACE