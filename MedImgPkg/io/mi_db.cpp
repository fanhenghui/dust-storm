#include "mi_db.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "mysql_connection.h"
#include "mi_io_logger.h"
#include "mi_md5.h"
#include "util/mi_memory_shield.h"

MED_IMG_BEGIN_NAMESPACE

const static std::string ROLE_TABLE = "role";
const static std::string USER_TABLE = "user";
const static std::string PATIENT_TABLE = "patient";
const static std::string STUDY_TABLE = "study";
const static std::string SERIES_TABLE = "series";
const static std::string INSTANCE_TABLE = "instance";
const static std::string PREPROCESS_TYPE_TABLE = "preprocess_type";
const static std::string PREPROCESS_TABLE = "preprocess";
const static std::string EVALUATION_TYPE_TABLE = "evaluation_type";
const static std::string EVALUATION_TABLE = "evaluation";
const static std::string ANNO_TABLE = "annotation";

#define TRY_CONNECT \
if (!this->try_connect()) {\
    MI_IO_LOG(MI_ERROR) << "db connection invalid.";\
    return -1;\
}

inline int get_patient_hash(const PatientInfo& patient_info, std::string& patient_hash) {
    std::stringstream patient_msg;
    patient_msg << patient_info.patient_id << "."
        << patient_info.patient_name << "."
        << patient_info.patient_birth_date;
    char md5[32] = { 0 };
    if( -1 == MD5::digest(patient_msg.str(), md5)) {
        MI_IO_LOG(MI_ERROR) << "calculate patient md5 failed.";
        return -1;
    }
    for(int i=0; i<32; ++i) {
        patient_hash.push_back(md5[i]);
    }
    return 0;
}

DB::DB() {}

DB::~DB() {
}

int DB::insert_patient(PatientInfo& patient_info) {
    TRY_CONNECT
    
    if (patient_info.md5.empty()) {
        if(-1 == get_patient_hash(patient_info, patient_info.md5)) {
            MI_IO_LOG(MI_ERROR) << "calcualte patient md5 failed.";
            return -1;
        }
    }
    std::stringstream sql;
    {
        sql << "INSERT INTO " << PATIENT_TABLE <<"(patient_id,patient_name,patient_sex,patient_birth_date,md5)" << " VALUES (\'"
        << patient_info.patient_id << "\',\'" 
        << patient_info.patient_name << "\',\'" 
        << patient_info.patient_sex << "\',\'" 
        << patient_info.patient_birth_date << "\',\'" 
        << patient_info.md5 << "\')";

        sql::ResultSet* res = nullptr;
        if(-1 == this->query(sql.str(), res) ) {
            StructShield<sql::ResultSet> shield(res);
            MI_IO_LOG(MI_ERROR) << "db insert patient failed.";
            return -1;
        } else {
            StructShield<sql::ResultSet> shield(res);
            if (res->next()) {
                patient_info.id = res->getInt64("id");
            }
        }
    }

    return 0;    
}

// int DB::insert_dcm_series(StudyInfo& study_info, SeriesInfo& series_info, PatientInfo& patient_info, UserInfo& user_info, 
//          const std::vector<DcmInstanceInfo>& instance_info) {
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     // insert new one
//     try {
//         //1 insert patient table
//         std::string patient_hash;
//         {
//             std::stringstream patient_msg;
//             patient_msg << patient_info.patient_id << "."
//             << patient_info.patient_name << "."
//             << patient_info.patient_birth_date;
//             char md5[32] = { 0 };
//             if( -1 == MD5::digest(patient_msg.str(), md5)) {
//                 MI_IO_LOG(MI_ERROR) << "calculate patient md5 failed.";
//                 return -1;
//             }
//             for(int i=0; i<32; +i) {
//                 patient_hash.push_back(md5[i]);
//             }
//         } 
//         PatientInfo patient_key;
//         patient_key.md5 = patient_hash;
//         std::vector<PatientInfo> patient_res;      
//         if( -1 == this->query_patient(patient_key, &patient_res)) {
//             MI_IO_LOG(MI_ERROR) << "query patient md5 failed.";
//             return -1;
//         }
//         int patient_fk = -1;
//         if (patient_res.empty()) {
//             // create new patient
//             patient_info.md5 = patient_hash;
//             if (-1 == this->insert_patient(patient_info)) {
//                 MI_IO_LOG(MI_ERROR) << "inert new patient failed.";
//                 return -1;
//             }
//         } else {
//             patient_info.id = patient_res[0].id;
//         }

//     //     study_info

//     //     // insert new item
//     //     std::string sql_str;
//     //     {
//     //         std::stringstream ss;
//     //         ss << "INSERT INTO " << DCM_TABLE << " (series_id, study_id, patient_name, "
//     //            "patient_id, modality, size_mb, dcm_path, preprocess_mask_path) values (";
//     //         ss << "\'" << item.series_id << "\',";
//     //         ss << "\'" << item.study_id << "\',";
//     //         ss << "\'" << item.patient_name << "\',";
//     //         ss << "\'" << item.patient_id << "\',";
//     //         ss << "\'" << item.modality << "\',";
//     //         ss << "\'" << item.size_mb << "\',";
//     //         ss << "\'" << item.dcm_path << "\',";
//     //         ss << "\'" << item.preprocess_mask_path << "\'";
//     //         ss << ");";
//     //         sql_str = ss.str();
//     //     }
//     //     sql::PreparedStatement* pstmt = _connection->prepareStatement(sql_str.c_str());
//     //     sql::ResultSet* res = pstmt->executeQuery();
//     //     delete pstmt;
//     //     pstmt = nullptr;
//     //     delete res;
//     //     res = nullptr;

//     // } catch (const sql::SQLException& e) {
//     //     MI_IO_LOG(MI_ERROR) << "qurey db when inset item failed with exception: "
//     //     << this->get_sql_exception_info(&e);
        
//     //     //TODO recovery old one if insert failed
//     //     return -1;
//     // }

    


//     return 0;
// }

// int DB::insert_evaluation(const EvaItem& eva) {

// }

// int DB::insert_annotation(const AnnoItem& anno) {

// }

// int DB::insert_preprocess(const PreprocessItem& prep) {

// }

// int DB::query_dicom(std::vector<DcmInfo>& dcm_infos, const QueryKey& key, QueryLevel query_level) {
//     MI_IO_LOG(MI_TRACE) << "IN db inset item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }
// }

// int DB::query_user(const std::string& user_name, int role) {
//     MI_IO_LOG(MI_TRACE) << "IN db query item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         sql::PreparedStatement* pstmt = nullptr;
//         sql::ResultSet* res = nullptr;

//         std::string sql_str;
//         {
//             std::stringstream ss;
//             ss << "SELECT name, role_fk FROM " << USER_TABLE << " WHERE " <<;
//             if (user_name.empty()) {
//                 ss << "name=\'" << user_name << "\'";    
//             }
//             if (role != -1) {
//                 ss << "role_fk=" << role;    
//             }
//             ss << ";";
//             sql_str = ss.str();
//         }
//         pstmt = _connection->prepareStatement(sql_str.c_str());
//         res = pstmt->executeQuery();
//         delete pstmt;
//         pstmt = nullptr;

//         if (res->next()) {
//             delete res;
//             res = nullptr;
//             in_db = true;
//         } else {
//             delete res;
//             res = nullptr;
//             in_db = false;
//         }
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db query item failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db query item.";
//     return 0;    
// }

// int DB::query_evaluation(int series_id, int eva_type) {

// }

// int DB::query_preprocess(int series_id, int prep_type) {

// }

// int DB::query_annotation(int series_id, int anno_type, const std::string& user_id) {

// }

// int DB::delete_dcm_series(int series_id) {

// }

// int DB::delete_evaluation(int eva_id) {

// }

// int DB::delete_annotation(int anno_id) {

// }

// int DB::delete_preprocess(int prep_id) {

// }

// int DB::modify_dcm_series(const DcmInfo& series_info, const std::vector<DcmInstanceInfo>& instance_info, const std::string& user) {

// }

// int DB::modify_evaluation(int eva_id, const EvaItem& eva) {

// }

// int DB::modify_annotation(int anno_id, const AnnoItem& anno) {

// }

// int DB::modify_preprocess(int prep_id, const PreprocessItem& prep) {

// }




// int DB::insert_dcm_item(const ImgItem& item) {
//     MI_IO_LOG(MI_TRACE) << "IN db inset item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     //query    
//     bool in_db(false);
//     if(-1 == query_dcm_item(item.series_id, in_db)) {
//         MI_IO_LOG(MI_ERROR) << "query failed when insert item.";
//         return -1;
//     }

//     //delete if exit
//     if (in_db) {
//         if(-1 == delete_dcm_item(item.series_id)) {
//             MI_IO_LOG(MI_ERROR) << "delete item failed when insert the item with the same primary key.";
//             return -1;
//         }
//     }

//     // insert new one
//     try {
//         // insert new item
//         std::string sql_str;
//         {
//             std::stringstream ss;
//             ss << "INSERT INTO " << DCM_TABLE << " (series_id, study_id, patient_name, "
//                "patient_id, modality, size_mb, dcm_path, preprocess_mask_path) values (";
//             ss << "\'" << item.series_id << "\',";
//             ss << "\'" << item.study_id << "\',";
//             ss << "\'" << item.patient_name << "\',";
//             ss << "\'" << item.patient_id << "\',";
//             ss << "\'" << item.modality << "\',";
//             ss << "\'" << item.size_mb << "\',";
//             ss << "\'" << item.dcm_path << "\',";
//             ss << "\'" << item.preprocess_mask_path << "\'";
//             ss << ");";
//             sql_str = ss.str();
//         }
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(sql_str.c_str());
//         sql::ResultSet* res = pstmt->executeQuery();
//         delete pstmt;
//         pstmt = nullptr;
//         delete res;
//         res = nullptr;

//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "qurey db when inset item failed with exception: "
//         << this->get_sql_exception_info(&e);
        
//         //TODO recovery old one if insert failed
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db query inset item."; 
//     return 0;
// }

// int DB::delete_dcm_item(const std::string& series_id) {
//     MI_IO_LOG(MI_TRACE) << "IN db query delete item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     //query    
//     bool in_db(false);
//     if(-1 == query_dcm_item(series_id, in_db)) {
//         MI_IO_LOG(MI_ERROR) << "query failed when insert item.";
//         return -1;
//     }

//     if (!in_db) {
//         MI_IO_LOG(MI_WARNING) << "delete item not exist which series is: " << series_id;        
//         return 0;
//     }

//     //delete
//     try {
//         std::string sql_str;
//         {
//             std::stringstream ss;
//             ss << "DELETE FROM " << DCM_TABLE << " where series_id=\'" << series_id << "\';";
//             sql_str = ss.str();
//         }
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(sql_str.c_str());
//         sql::ResultSet* res = pstmt->executeQuery();
//         delete pstmt;
//         pstmt = nullptr;
//         delete res;
//         res = nullptr;

//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "qurey db when inset item failed with exception: "
//         << this->get_sql_exception_info(&e);
//         // TODO recovery DB if delete failed
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db delete item."; 
//     return 0;
// }

// int DB::query_dcm_item(const std::string& series_id, bool& in_db) {
//     MI_IO_LOG(MI_TRACE) << "IN db query item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         sql::PreparedStatement* pstmt = nullptr;
//         sql::ResultSet* res = nullptr;

//         std::string sql_str;
//         {
//             std::stringstream ss;
//             ss << "SELECT * FROM " << DCM_TABLE << " where series_id=\'" << series_id << "\';";
//             sql_str = ss.str();
//         }
//         pstmt = _connection->prepareStatement(sql_str.c_str());
//         res = pstmt->executeQuery();
//         delete pstmt;
//         pstmt = nullptr;

//         if (res->next()) {
//             delete res;
//             res = nullptr;
//             in_db = true;
//         } else {
//             delete res;
//             res = nullptr;
//             in_db = false;
//         }
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db query item failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db query item.";
//     return 0;    
// }

// int DB::get_dcm_item(const std::string& series_id, ImgItem& item) {
//     MI_IO_LOG(MI_TRACE) << "IN db get item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << DCM_TABLE << " WHERE series_id=\'" << series_id << "\';";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         if (res->next()) {
//             item.series_id = res->getString("series_id");
//             item.study_id = res->getString("study_id");
//             item.patient_name = res->getString("patient_name");
//             item.patient_id = res->getString("patient_id");
//             item.modality = res->getString("modality");
//             item.dcm_path = res->getString("dcm_path");
//             item.preprocess_mask_path = res->getString("preprocess_mask_path");
//             item.annotation_ai_path = res->getString("annotation_ai_path");
//             item.size_mb = res->getInt("size_mb");
//             item.ai_intermediate_data_path = res->getString("ai_intermediate_data_path");
//             delete res;
//             res = nullptr;
//             delete pstmt;
//             pstmt = nullptr;
//         } else {
//             delete res;
//             res = nullptr;
//             delete pstmt;
//             pstmt = nullptr;
//             return -1;
//         }
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get item failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }

//     MI_IO_LOG(MI_TRACE) << "Out db get item."; 
//     return 0; 
// }

// int DB::get_ai_annotation_item(const std::string& series_id, std::string& annotation_ai_path) {
//     MI_IO_LOG(MI_TRACE) << "IN db get AI annotation item."; 
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << DCM_TABLE << " WHERE series_id=\'" << series_id << "\';";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         if (res->next()) {
//             annotation_ai_path = res->getString("annotation_ai_path");
//             delete res;
//             res = nullptr;
//             delete pstmt;
//             pstmt = nullptr;
//         } else {
//             delete res;
//             res = nullptr;
//             delete pstmt;
//             pstmt = nullptr;
//             return -1;
//         }
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get item failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }

//     MI_IO_LOG(MI_TRACE) << "Out db get AI annotation item."; 
//     return 0; 
// }

// int DB::get_usr_annotation_items_by_series(const std::string& series_id, std::vector<AnnoItem>& items) {
//     MI_IO_LOG(MI_TRACE) << "IN db get usr annotation items by series id.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << ANNO_TABLE << " WHERE series_id=\'" << series_id << "\'";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         items.clear();
//         while(true) {
//             if(res->next()) {
//                 AnnoItem item;
//                 item.series_id = series_id;
//                 item.usr_name = res->getString("usr_name");
//                 item.annotation_usr_path = res->getString("annotation_usr_path");
//                 items.push_back(item);
//             } else {
//                 break;
//             }
//         }
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get annotation items by series id failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db get usr annotation items by series id.";
//     return 0;
// }

// int DB::get_usr_annotation_items_by_usr(const std::string& usr_name, std::vector<AnnoItem>& items) {
//     MI_IO_LOG(MI_TRACE) << "IN db get usr annotation items by usr.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << ANNO_TABLE << " WHERE usr_name=\'" << usr_name << "\'";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         items.clear();
//         while(true) {
//             if(res->next()) {
//                 AnnoItem item;
//                 item.series_id = res->getString("series_id");
//                 item.usr_name = usr_name;
//                 item.annotation_usr_path = res->getString("annotation_usr_path");
//                 items.push_back(item);
//             } else {
//                 break;
//             }
//         }
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get annotation items by usr name failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db get usr annotation items by usr.";
//     return 0;
// }

// int DB::get_usr_annotation_item(const std::string& series_id, const std::string& usr_name, std::vector<DB::AnnoItem>& items) {
//     MI_IO_LOG(MI_TRACE) << "IN db get usr annotation item.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << ANNO_TABLE << " WHERE usr_name=\'" << usr_name << "\'" 
//         << " and series_id=\'" << series_id << "\'";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         items.clear();
//         while(true) {
//             if(res->next()) {
//                 AnnoItem item;
//                 item.series_id = series_id;
//                 item.usr_name = usr_name;
//                 item.annotation_usr_path = res->getString("annotation_usr_path");
//                 items.push_back(item);
//             } else {
//                 break;
//             }
//         }
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get annotation item failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db get usr annotation item.";
//     return 0;
// }

// int DB::get_all_dcm_items(std::vector<ImgItem>& items) {
//     MI_IO_LOG(MI_TRACE) << "IN db get all dcm items."; 
//     if (!this->try_connect()) {;
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << DCM_TABLE;
//         sql::PreparedStatement* pstmt =
//             _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         items.clear();
//         std::cout << res->rowsCount() << std::endl;

//         while (true) {
//             if (res->next()) {
//                 ImgItem item;
//                 item.series_id = res->getString("series_id");
//                 item.study_id = res->getString("study_id");
//                 item.patient_name = res->getString("patient_name");
//                 item.patient_id = res->getString("patient_id");
//                 item.modality = res->getString("modality");
//                 item.dcm_path = res->getString("dcm_path");
//                 item.preprocess_mask_path = res->getString("preprocess_mask_path");
//                 item.annotation_ai_path = res->getString("annotation_ai_path");
//                 item.size_mb = res->getInt("size_mb");
//                 item.ai_intermediate_data_path = res->getInt("ai_intermediate_data_path");
//                 items.push_back(item);
//             } else {
//                 break;
//             }
//         }
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get all items failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db get all dcm items."; 
//     return 0;
// }

// int DB::get_all_usr_annotation_items(std::vector<AnnoItem>& items) {
//     MI_IO_LOG(MI_TRACE) << "IN db get all usr annotation items.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "SELECT * FROM " << ANNO_TABLE;
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();

//         items.clear();
//         while(true) {
//             if(res->next()) {
//                 AnnoItem item;
//                 item.series_id = res->getString("series_id");
//                 item.usr_name = res->getString("usr_name");
//                 item.annotation_usr_path = res->getString("annotation_usr_path");
//                 items.push_back(item);
//             } else {
//                 break;
//             }
//         }
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db get all usr annotation items failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db get all usr annotation items.";
//     return 0;
// }

// int DB::update_preprocess_mask(const std::string& series_id, const std::string& preprocess_mask_path) {
//     MI_IO_LOG(MI_TRACE) << "IN db update preprocess mask.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "UPDATE " << DCM_TABLE << " SET preprocess_mask_path=\'" << preprocess_mask_path << "\'" 
//         << " WHERE series_id=\'" << series_id << "\'";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db update preprocess mask failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db update preprocess mask.";
//     return 0;
// }

// int DB::update_ai_annotation(const std::string& series_id, const std::string& annotation_ai_path) {
//     MI_IO_LOG(MI_TRACE) << "IN db update AI annotation.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "UPDATE " << DCM_TABLE << " SET annotation_ai_path=\'" << annotation_ai_path << "\'" 
//         << " WHERE series_id=\'" << series_id << "\'";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db update AI annotation failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db update AI annotation.";
//     return 0;
// }

// int DB::update_ai_intermediate_data(const std::string& series_id, const std::string& ai_intermediate_data_path) {
//     MI_IO_LOG(MI_TRACE) << "IN db update AI intermediate data.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "UPDATE " << DCM_TABLE << " SET ai_intermediate_data_path=\'" << ai_intermediate_data_path << "\'" 
//         << " WHERE series_id=\'" << series_id << "\'";
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db update AI intermediate data failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db update AI intermediate data.";
//     return 0;
// }

// int DB::update_usr_annotation(const std::string& series_id, const std::string& usr_name, const std::string& annotation_usr_path) {
//     MI_IO_LOG(MI_TRACE) << "IN db update usr annotation.";
//     if (!this->try_connect()) {
//         MI_IO_LOG(MI_ERROR) << "db connection invalid.";
//         return -1;
//     }

//     try {
//         std::stringstream ss;
//         ss << "UPDATE " << ANNO_TABLE << " SET annotation_usr_path=\'" << annotation_usr_path << "\'" 
//         << " WHERE series_id=\'" << series_id << "\'" << " AND usr_name=\'" << usr_name;
//         sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
//         sql::ResultSet* res = pstmt->executeQuery();
//         delete res;
//         res = nullptr;
//         delete pstmt;
//         pstmt = nullptr;
//     } catch (const sql::SQLException& e) {
//         MI_IO_LOG(MI_ERROR) << "db update usr annotation failed with exception: "
//         << this->get_sql_exception_info(&e);
//         return -1;
//     }
//     MI_IO_LOG(MI_TRACE) << "OUT db update usr annotation.";
//     return 0;
// }

MED_IMG_END_NAMESPACE