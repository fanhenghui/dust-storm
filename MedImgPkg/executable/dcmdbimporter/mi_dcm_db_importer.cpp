#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_cache_db.h"
#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "util/mi_file_util.h"

using namespace medical_imaging;

static const std::string CACHE_DB_NAMA = "med_img_cache_db";
static const std::string DB_NAME = "med_img_db";
int _db_type = 0;//0 for db 1 for cache_db

static std::shared_ptr<MySQLDB> _db;

struct DcmItem {
    std::string series_id;
    std::string study_id;
    std::string patient_name;
    std::string patient_id;
    std::string modality;
};

static std::ofstream out_log;

class LogSheild {
public:
    LogSheild() {
        out_log.open("db_importer.log", std::ios::out);

        if (out_log.is_open()) {
            out_log << "DB importer log:\n";
        }
    }
    ~LogSheild() {
        out_log.close();
    }
};
#define LOG_OUT(info)                                                          \
  std::cout << info;                                                           \
  out_log << info;

int connect_db(const std::string& user, const std::string& ip_port, const std::string& pwd, const std::string& db) {
    if (0 == _db_type) {
        _db.reset(new DB());
        return _db->connect(user, ip_port, pwd, db);
    } else {
        _db.reset(new CacheDB());
        return _db->connect(user, ip_port, pwd, db);   
    }
}

int parse_root_dcm(
    std::string& root,
    std::map<std::string, std::vector<std::string>>& study_series_col,
    std::map<std::string, DcmItem>& series_info_map,
    std::map<std::string, std::vector<std::string>>& series_col) {
    std::vector<std::string> files;
    std::set<std::string> postfix;
    postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(root, postfix, files);

    if (files.empty()) {
        LOG_OUT("ERROR : has no .dcm files.");
        return -1;
    }

    study_series_col.clear();
    series_info_map.clear();
    series_col.clear();
    DICOMLoader loader;

    for (size_t i = 0; i < files.size(); i++) {
        std::string series_id;
        std::string study_id;
        std::string patient_name;
        std::string patient_id;
        std::string modality;

        if (IO_SUCCESS ==
                loader.check_series_uid(files[i], study_id, series_id, patient_name,
                                        patient_id, modality)) {
            if (series_info_map.find(series_id) == series_info_map.end()) {
                DcmItem info;
                info.series_id = series_id;
                info.study_id = study_id;
                info.patient_name = patient_name;
                info.patient_id = patient_id;
                info.modality = modality;
                series_info_map[series_id] = info;
            } else if (series_info_map[series_id].study_id != study_id) {
                std::stringstream ss;
                ss << "ERROR : series study conflict " << files[i] << "\n";
                LOG_OUT(ss.str());
                return -1;
            }

            auto it = series_col.find(series_id);

            if (it == series_col.end()) {
                series_col.insert(
                    std::make_pair(series_id, std::vector<std::string>(1, files[i])));
            } else {
                it->second.push_back(files[i]);
            }

            it = study_series_col.find(study_id);

            if (it == study_series_col.end()) {
                study_series_col.insert(
                    std::make_pair(study_id, std::vector<std::string>(1, series_id)));
            } else {
                it->second.push_back(series_id);
            }

        } else {
            std::stringstream ss;
            ss << "ERROR : read file " << files[i] << " failed, skip it.\n";
            LOG_OUT(ss.str());
        }
    }

    return 0;
}

int copy_file(const std::string& src, const std::string& dst, float& size_mb) {
    size_mb = 0;
    if (src.empty()) {
        LOG_OUT("ERROR : src path is empty.");
        return -1;
    }

    if (dst.empty()) {
        LOG_OUT("ERROR : dst path is empty.");
        return -1;
    }

    std::ifstream in;
    std::ofstream out;
    in.open(src, std::ios::binary | std::ios::in);
    out.open(dst, std::ios::binary | std::ios::out);

    if (!in.is_open()) {
        LOG_OUT("ERROR : open src file failed.");
        in.close();
        out.close();
        return -1;
    }

    if (!out.is_open()) {
        LOG_OUT("ERROR : open dst file failed.");
        in.close();
        out.close();
        return -1;
    }

    // cal size
    in.seekg(0, std::ios::end);
    size_mb = static_cast<float>(in.tellg())/1024.0/1024.0;
    in.seekg(0, std::ios::beg);
    // copy
    out << in.rdbuf();


    in.close();
    out.close();

    return 0;
}

int create_dcm_folder(
    std::string map_path,
    std::map<std::string, std::vector<std::string>>& study_series_col) {
    for (auto study = study_series_col.begin(); study != study_series_col.end();
            ++study) {
        const std::string study_path = map_path + "/" + study->first;

        if (-1 == access(study_path.c_str(), F_OK)) {
            mkdir(study_path.c_str(), S_IRWXU);
        }

        for (auto series = study->second.begin(); series != study->second.end();
                ++series) {
            const std::string series_path = study_path + "/" + *series;

            if (-1 == access(series_path.c_str(), F_OK)) {
                mkdir(series_path.c_str(), S_IRWXU);
            }
        }
    }

    return 0;
}

int import_one_series_cache(const std::string& map_path, DcmItem& series_info,
                      const std::string& series,
                      const std::vector<std::string>& files) {
    // copy src file to dst map
    const std::string series_dst =
        map_path + "/" + series_info.study_id + "/" + series;
    CacheDB::ImgItem item;
    item.path = series_dst;
    item.series_id = series_info.series_id;
    item.study_id = series_info.study_id;
    item.modality = series_info.modality;
    item.patient_id = series_info.patient_id;
    item.patient_name = series_info.patient_name;
    float size_mb = 0;

    for (size_t i = 0; i < files.size(); i++) {
        boost::filesystem::path p(files[i].c_str());
        std::string base_name = p.filename().string();
        const std::string file_dst = series_dst + "/" + base_name;
        float size_slice = 0;
        if (0 != copy_file(files[i], file_dst, size_slice)) {
            std::stringstream ss;
            ss << "copy file " << base_name << " failed.\n";
            LOG_OUT(ss.str());
            return -1;
        }
        size_mb += size_slice;
    }
    item.size_mb = size_mb;

    std::shared_ptr<CacheDB> db = std::dynamic_pointer_cast<CacheDB>(_db);

    return db->insert_item(item);
};

int import_cache_db(std::string map_path,
              std::map<std::string, DcmItem>& series_info_map,
              std::map<std::string, std::vector<std::string>>& series_col) {

    for (auto it = series_col.begin(); it != series_col.end(); it++) {
        const std::string series = it->first;
        DcmItem& series_info = series_info_map[series];

        if (-1 == import_one_series_cache(map_path, series_info, series, it->second)) {
            std::stringstream ss;
            ss << "ERROR : import series " << series << "failed, skip it.\n";
            LOG_OUT(ss.str());
        } else {
            std::stringstream ss;
            ss << "import series : " << series << " done.\n";
            LOG_OUT(ss.str());
        }
    }

    return 0;
}

int import_one_series(const std::string& map_path, DcmItem& series_info,
                      const std::string& series,
                      const std::vector<std::string>& files) {
    // copy src file to dst map
    const std::string series_dst =
        map_path + "/" + series_info.study_id + "/" + series;
    DB::ImgItem item;
    item.dcm_path = series_dst;
    item.series_id = series_info.series_id;
    item.study_id = series_info.study_id;
    item.modality = series_info.modality;
    item.patient_id = series_info.patient_id;
    item.patient_name = series_info.patient_name;
    float size_mb = 0;

    for (size_t i = 0; i < files.size(); i++) {
        boost::filesystem::path p(files[i].c_str());
        std::string base_name = p.filename().string();
        const std::string file_dst = series_dst + "/" + base_name;
        float size_slice = 0;
        if (0 != copy_file(files[i], file_dst, size_slice)) {
            std::stringstream ss;
            ss << "copy file " << base_name << " failed.\n";
            LOG_OUT(ss.str());
            return -1;
        }
        size_mb += size_slice;
    }
    item.size_mb = size_mb;

    std::shared_ptr<DB> db = std::dynamic_pointer_cast<DB>(_db);

    return db->insert_dcm_item(item);
};

int import_db(std::string map_path,
              std::map<std::string, DcmItem>& series_info_map,
              std::map<std::string, std::vector<std::string>>& series_col) {

    for (auto it = series_col.begin(); it != series_col.end(); it++) {
        const std::string series = it->first;
        DcmItem& series_info = series_info_map[series];

        if (-1 == import_one_series(map_path, series_info, series, it->second)) {
            std::stringstream ss;
            ss << "ERROR : import series " << series << "failed, skip it.\n";
            LOG_OUT(ss.str());
        } else {
            std::stringstream ss;
            ss << "import series : " << series << " done.\n";
            LOG_OUT(ss.str());
        }
    }

    return 0;
}

int import_rle(std::string root) {
    std::set<std::string> postfix;
    postfix.insert(".rle");
    std::vector<std::string> files;
    FileUtil::get_all_file_recursion(root, postfix, files);
    if (files.empty()) {
        std::stringstream ss;
        ss << "find no rle files.\n";
        LOG_OUT(ss.str());
        return -1;
    }

    std::shared_ptr<DB> db = std::dynamic_pointer_cast<DB>(_db);

    for (auto it = files.begin(); it != files.end(); ++it) {
        std::string file = *it;
        boost::filesystem::path path(*it);
        const std::string base_name = path.filename().string();
        int idx = base_name.size()-1;
        std::string series_id;
        for(idx = base_name.size()-1; idx >=0 ; --idx) {
            if (base_name[idx] == '.') {
                series_id = base_name.substr(0,idx);
                break;
            }    
        }
        //move file to path
        DB::ImgItem item;
        if(0 == db->get_dcm_item(series_id, item) ) {
            const std::string dcm_path = item.dcm_path;
            float mb_size = 0;
            copy_file(file, dcm_path+"/" + base_name, mb_size);
            db->update_preprocess_mask(series_id, dcm_path+"/" + base_name);
            std::stringstream ss;
            ss << "update preprocessing mask file: " << base_name << ".\n";
            LOG_OUT(ss.str());
        } else {
            std::stringstream ss;
            ss << "cant find series: " << series_id << " in db.\n";
            LOG_OUT(ss.str());
        }
    }

    return 0;
}

int import_csv(std::string root) {
    std::set<std::string> postfix;
    postfix.insert(".csv");
    std::vector<std::string> files;
    FileUtil::get_all_file_recursion(root, postfix, files);
    if (files.empty()) {
        std::stringstream ss;
        ss << "find no rle files.\n";
        LOG_OUT(ss.str());
        return -1;
    }

    std::shared_ptr<DB> db = std::dynamic_pointer_cast<DB>(_db);

    for (auto it = files.begin(); it != files.end(); ++it) {
        std::string file = *it;
        boost::filesystem::path path(*it);
        const std::string base_name = path.filename().string();
        int idx = base_name.size()-1;
        std::string series_id;
        for(idx = base_name.size()-1; idx >=0 ; --idx) {
            if (base_name[idx] == '.') {
                series_id = base_name.substr(0,idx);
                break;
            }    
        }
        //move file to path
        DB::ImgItem item;
        if(0 == db->get_dcm_item(series_id, item) ) {
            const std::string dcm_path = item.dcm_path;
            float mb_size = 0;
            copy_file(file, dcm_path+"/" + base_name, mb_size);
            db->update_ai_annotation(series_id, dcm_path+"/" + base_name);
            std::stringstream ss;
            ss << "update AI annotation file: " << base_name << ".\n";
            LOG_OUT(ss.str());
        } else {
            std::stringstream ss;
            ss << "cant find series: " << series_id << " in db.\n";
            LOG_OUT(ss.str());
        }
    }

    return 0;
}

int getch() {
    struct termios tm, tm_old;
    int fd = 0;
    int c = 0;

    if (tcgetattr(fd, &tm) < 0) {
        return -1;
    }

    tm_old = tm; // save original mode
    cfmakeraw(&tm);

    if (tcsetattr(fd, TCSANOW, &tm) < 0) { // set new mode
        return -1;
    }

    c = fgetc(stdin);

    if (tcsetattr(fd, TCSANOW, &tm_old) < 0) { // set old mode
        return -1;
    }

    return c;
}

void print_h() {
    printf("DICOM DB importer:\n");
    printf("\t-u : mysql login in user {root}.\n");
    printf("\t-i : mysql login in ip&port {127.0.0.1:3306}.\n");
    printf("\t-d : DB name {med_img_cache_db & med_img_db}.\n");
    printf("\t-r : import data root.\n");
    printf("\t-m : DB map path.\n");
    printf("\t-h : help.\n");
}

int main(int argc, char* argv[]) {

    LogSheild log;

    std::string user;
    std::string db;
    std::string root;
    std::string ip = "127.0.0.1";
    std::string map_path;
    bool is_rle = true;
    bool is_csv = true;
    int ch;

    while ((ch = getopt(argc, argv, "cahu:p:d:r:i:m:")) != -1) {
        switch (ch) {
        case 'u':
            user = std::string(optarg);
            break;

        case 'd':
            db = std::string(optarg);
            break;

        case 'r':
            root = std::string(optarg);
            break;

        case 'i':
            ip = std::string(optarg);
            break;

        case 'm':
            map_path = std::string(optarg);
            break;

        case 'h':
            print_h();
            return 0;
            break;

        case 'c':
            is_rle = true;
            break;

        case 'a':
            is_csv = true;
            break;

        default:
            break;
        }
    }

    if (user.empty()) {
        LOG_OUT("ERROR : mysql user is empty.\n");
        return -1;
    }

    if (db.empty()) {
        LOG_OUT("ERROR : db name is empty.\n");
        return -1;
    }

    if (root.empty()) {
        LOG_OUT("ERROR :　data root is empty.\n");
        return -1;
    }

    if (ip.empty()) {
        LOG_OUT("ERROR : db ip&port is empty.\n");
        return -1;
    }

    if (map_path.empty()) {
        LOG_OUT("ERROR : db map path is empty.\n");
        return -1;
    }

    printf("please enter %s's password:", user.c_str());
    std::string pwd;
    char c;

    while ((c = getch()) != '\r') {
        if (c == 8 && pwd.size() > 0) {
            pwd.pop_back();
        }

        pwd.push_back(c);
    }

    if(db == CACHE_DB_NAMA) {
        _db_type = 1;
    } else {
        _db_type = 0;
    }
    if (0 != connect_db(user, ip, pwd, db)) {
        LOG_OUT("ERROR : connect DB failed.");
        return -1;
    }

    if (_db_type == 1) {
        std::map<std::string, std::vector<std::string>> study_series_col;
        std::map<std::string, DcmItem> series_info_map;
        std::map<std::string, std::vector<std::string>> series_col;

        if (0 != parse_root_dcm(root, study_series_col, series_info_map, series_col)) {
            LOG_OUT("ERROR : parse root failed.");
            return -1;
        }
    
        if (0 != create_dcm_folder(map_path, study_series_col)) {
            LOG_OUT("ERROR : create folder failed.");
            return -1;
        }

    
        if (0 != import_cache_db(map_path, series_info_map, series_col)) {
            LOG_OUT("ERROR : import DB failed.");
            return -1;
        }
    } else {
        std::map<std::string, std::vector<std::string>> study_series_col;
        std::map<std::string, DcmItem> series_info_map;
        std::map<std::string, std::vector<std::string>> series_col;

        //dcm
        if (0 != parse_root_dcm(root, study_series_col, series_info_map, series_col)) {
            LOG_OUT("ERROR : parse root failed.");
            return -1;
        }
        if (0 != create_dcm_folder(map_path, study_series_col)) {
            LOG_OUT("ERROR : create folder failed.");
            return -1;
        }    
        if (0 != import_db(map_path, series_info_map, series_col)) {
            LOG_OUT("ERROR : import DB failed.");
            return -1;
        }

        if (is_rle) {
            //import rle
            import_rle(root);
        }

        if (is_csv) {
            import_csv(root);
        }

        //mask
        //preprocess mask(series.ppm.rle)
        //lung mask(series.aim.rle)
        
        //annotation file
        //AI annotation series.ai.dcsv
        //usr annotation series.usr.usr_name.dcsv
    }

    // test
    // std::vector<DcmItem> items;

    // if (0 == data_base.get_all_item(items)) {
    //     for (size_t i = 0; i < items.size(); ++i) {
    //         printf("series : %s ; study : %s ; patient_name : % s ; patient_id : %s "
    //                "; modality : % s\n",
    //                items[i].series_id.c_str(), items[i].study_id.c_str(),
    //                items[i].patient_name.c_str(), items[i].patient_id.c_str(),
    //                items[i].modality.c_str());
    //     }

    //     // std::string path;

    //     // if (0 == data_base.get_series_path(items[0].series_id, path)) {
    //     //     printf("get items 0 path : %s\n", path.c_str());
    //     // }
    // } else {
    //     LOG_OUT("ERROR : get all item failed.\n");
    // }
    _db->disconnect();
    LOG_OUT("import DB done.");
    return 0;
}