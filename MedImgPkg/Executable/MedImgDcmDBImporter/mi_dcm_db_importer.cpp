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
#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "mysql_connection.h"
// mysql end

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgUtil/mi_file_util.h"

using namespace medical_imaging;

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


struct SeriesInfo{
  std::string study_id;
  std::string patient_name;
  std::string patient_id;
  std::string modality;
};

int connect_db(sql::Connection *&con, const std::string &user,
               const std::string &ip_port, const std::string &pwd,
               const std::string &db) {
  con = nullptr;
  try {
    // create connect
    sql::Driver *driver = get_driver_instance();
    con = driver->connect(ip_port.c_str(), user.c_str(), pwd.c_str());
    // con = driver->connect("tcp://127.0.0.1:3306", "root", "0123456");
    con->setSchema(db.c_str());
    return 0;
  } catch (const sql::SQLException &e) {
    std::stringstream ss;
    ss << "ERROR : ";
    ss << "# ERR: SQLException in " << __FILE__;
    ss << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
    ss << "# ERR: " << e.what();
    ss << " (MySQL error code: " << e.getErrorCode();
    ss << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    LOG_OUT(ss.str());
    delete con;
    con = nullptr;

    return -1;
  }
}

int parse_root(
    std::string &root,
    std::map<std::string, std::vector<std::string>> &study_series_col,
    std::map<std::string, SeriesInfo> &series_info_map,
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
        SeriesInfo info;
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

int import_one_series(sql::Connection *con, const std::string &map_path,
                      const SeriesInfo &series_info, const std::string &series,
                      const std::vector<std::string> &files) {
  // copy src file to dst map
  const std::string series_dst = map_path + "/" + series_info.study_id + "/" + series;
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

  // write to DB (find ; delete if exit ; insert)
  try {
    sql::Statement *stmt = con->createStatement();
    delete stmt;
    sql::PreparedStatement *pstmt = nullptr;
    sql::ResultSet *res = nullptr;

    // find
    std::string sql_str;
    {
      std::stringstream ss;
      ss << "SELECT * FROM img_tbl where series_id=\'" << series << "\';";
      sql_str = ss.str();
    }
    pstmt = con->prepareStatement(sql_str.c_str());
    res = pstmt->executeQuery();
    delete pstmt;
    pstmt = nullptr;

    // delete if exit
    if (res->next()) {
      std::stringstream ss;
      ss << "WARNING : already has the same series item : " << series
         << " , use the new one replace it.\n";
      delete res;
      res = nullptr;

      // delete old one
      {
        std::stringstream ss;
        ss << "DELETE FROM img_tbl where series_id=\'" << series << "\';";
        sql_str = ss.str();
      }
      sql::PreparedStatement *pstmt = con->prepareStatement(sql_str.c_str());
      sql::ResultSet *res = pstmt->executeQuery();
      delete pstmt;
      pstmt = nullptr;
      delete res;
      res = nullptr;
    }

    // insert new item
    {
      std::stringstream ss;
      ss << "INSERT INTO img_tbl (series_id , study_id , patient_name , "
            "patient_id ,modality , path) values (";
      ss << "\'" << series << "\',";
      ss << "\'" << series_info.study_id << "\',";
      ss << "\'" << series_info.patient_name << "\',";
      ss << "\'" << series_info.patient_id << "\',";
      ss << "\'" << series_info.modality << "\',";
      ss << "\'" << series_dst << "\'";
      ss << ");";
      sql_str = ss.str();
    }
    pstmt = con->prepareStatement(sql_str.c_str());
    res = pstmt->executeQuery();
    delete pstmt;
    pstmt = nullptr;
    delete res;
    res = nullptr;

  } catch (const sql::SQLException &e) {
    out_log << "ERROR : ";
    out_log << "# ERR: SQLException in " << __FILE__;
    out_log << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
    out_log << "# ERR: " << e.what();
    out_log << " (MySQL error code: " << e.getErrorCode();
    out_log << ", SQLState: " << e.getSQLState() << " )" << std::endl;

    delete con;
    con = nullptr;

    // TODO recovery DB

    return -1;
  }

  return 0;
};

int import_db(sql::Connection *con, std::string map_path,
              std::map<std::string, SeriesInfo> &series_info_map,
              std::map<std::string, std::vector<std::string>> &series_col) {

  for (auto it = series_col.begin(); it != series_col.end(); it++) {
    const std::string series = it->first;
    const SeriesInfo &series_info = series_info_map[series];
    if (-1 == import_one_series(con, map_path, series_info, series, it->second)) {
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
  std::map<std::string, SeriesInfo> series_info_map;
  std::map<std::string, std::vector<std::string>> series_col;

  if (0 != parse_root(root, study_series_col, series_info_map, series_col)) {
    LOG_OUT("ERROR : parse root failed.");
    return -1;
  }

  if (0 != create_folder(map_path, study_series_col)) {
    LOG_OUT("ERROR : create folder failed.");
    return -1;
  }

  sql::Connection *con = nullptr;
  if (0 != connect_db(con, user, "tcp://" + ip, pwd, db)) {
    LOG_OUT("ERROR : connect DB failed.");
    return -1;
  }

  if (0 != import_db(con, map_path, series_info_map, series_col)) {
    LOG_OUT("ERROR : import DB failed.");
    return -1;
  }

  return 0;
}