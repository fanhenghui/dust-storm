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

// mysql begin
// #include "cppconn/driver.h"
// #include "cppconn/exception.h"
// #include "cppconn/prepared_statement.h"
// #include "cppconn/resultset.h"
// #include "cppconn/sqlstring.h"
// #include "cppconn/statement.h"
// #include "mysql_connection.h"
// mysql end

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "util/mi_file_util.h"
#include "MedImgAppCommon/mi_app_data_base.h"

using namespace medical_imaging;

AppDataBase data_base;

static std::ofstream out_log;
class LogSheild {
public:
  LogSheild() {
    out_log.open("db_importer.log", std::ios::out);
    if (out_log.is_open()) {
      out_log << "DB importer log:\n";
    }
  }
  ~LogSheild() { out_log.close(); }
};
#define LOG_OUT(info)                                                          \
  std::cout << info;                                                           \
  out_log << info;

int connect_db(const std::string &user,
               const std::string &ip_port, const std::string &pwd,
               const std::string &db) {
  return data_base.connect(user , ip_port , pwd , db);
}

int parse_root(
    std::string &root,
    std::map<std::string, std::vector<std::string>> &study_series_col,
    std::map<std::string, ImgItem> &series_info_map,
    std::map<std::string, std::vector<std::string>> &series_col) {
  std::vector<std::string> files;
  FileUtil::get_all_file_recursion(root, std::vector<std::string>(1, ".dcm"),
                                   files);
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
    if (IO_SUCCESS == loader.check_series_uid(files[i], study_id, 
      series_id, patient_name, patient_id, modality)) {
      if (series_info_map.find(series_id) == series_info_map.end()) {
        ImgItem info;
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

int copy_file(const std::string &src, const std::string &dst) {
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

  // copy
  out << in.rdbuf();

  in.close();
  out.close();

  return 0;
}

int create_folder(
    std::string map_path,
    std::map<std::string, std::vector<std::string>> &study_series_col) {
  for (auto study = study_series_col.begin(); study != study_series_col.end(); ++study) {
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

int import_one_series(const std::string &map_path,
                      ImgItem &series_info, const std::string &series,
                      const std::vector<std::string> &files) {
  // copy src file to dst map
  const std::string series_dst = map_path + "/" + series_info.study_id + "/" + series;
  series_info.path = series_dst;
  for (size_t i = 0; i < files.size(); i++) {
    boost::filesystem::path p(files[i].c_str());
    std::string base_name = p.filename().string();
    const std::string file_dst = series_dst + "/" + base_name;
    if (0 != copy_file(files[i], file_dst)) {
      std::stringstream ss;
      ss << "copy file " << base_name << " failed.\n";
      LOG_OUT(ss.str());
      return -1;
    }
  }

  return data_base.insert_item(series_info);
};

int import_db(std::string map_path,
              std::map<std::string, ImgItem> &series_info_map,
              std::map<std::string, std::vector<std::string>> &series_col) {

  for (auto it = series_col.begin(); it != series_col.end(); it++) {
    const std::string series = it->first;
    ImgItem &series_info = series_info_map[series];
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
  printf("\t-d : DB name {med_img_cache_db}.\n");
  printf("\t-r : import data root.\n");
  printf("\t-m : DB map path.\n");
  printf("\t-h : help.\n");
}

int main(int argc, char *argv[]) {

  LogSheild log;

  std::string user;
  std::string db;
  std::string root;
  std::string ip = "127.0.0.1";
  std::string map_path;
  int ch;
  while ((ch = getopt(argc, argv, "hu:p:d:r:i:m:")) != -1) {
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

  std::map<std::string, std::vector<std::string>> study_series_col;
  std::map<std::string, ImgItem> series_info_map;
  std::map<std::string, std::vector<std::string>> series_col;

  if (0 != parse_root(root, study_series_col, series_info_map, series_col)) {
    LOG_OUT("ERROR : parse root failed.");
    return -1;
  }

  if (0 != create_folder(map_path, study_series_col)) {
    LOG_OUT("ERROR : create folder failed.");
    return -1;
  }

  if (0 != connect_db(user, "tcp://" + ip, pwd, db)) {
    LOG_OUT("ERROR : connect DB failed.");
    return -1;
  }

  if (0 != import_db(map_path, series_info_map, series_col)) {
    LOG_OUT("ERROR : import DB failed.");
    return -1;
  }

  //test
  std::vector<ImgItem> items;
  if (0 == data_base.get_all_item(items)) {
    for (size_t i = 0; i < items.size(); ++i) {
      printf("series : %s ; study : %s ; patient_name : % s ; patient_id : %s "
             "; modality : % s\n",
             items[i].series_id.c_str(), items[i].study_id.c_str(),
             items[i].patient_name.c_str(), items[i].patient_id.c_str(),
             items[i].modality.c_str());
    }
    std::string path;
    if (0 == data_base.get_series_path(items[0].series_id, path)) {
      printf("get items 0 path : %s\n", path.c_str());
    }
  } else {
    LOG_OUT("ERROR : get all item failed.\n");
  }

  return 0;
}