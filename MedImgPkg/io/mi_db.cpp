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
#include "util/mi_time_util.h"

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

const static int UID_LIMIT = 64;
const static int DESCRIPTION_LIMIT = 64;

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
    if( 0 != MD5::digest(patient_msg.str(), md5)) {
        MI_IO_LOG(MI_ERROR) << "calculate patient md5 failed.";
        return -1;
    }
    for(int i=0; i<32; ++i) {
        patient_hash.push_back(md5[i]);
    }
    return 0;
}

DB::DB() {}

DB::~DB() {}

int DB::verfiy_study_info(const StudyInfo& study_info) {
    //verify
    //patient_fk
    if (study_info.patient_fk <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid patient fk: patient fk: " << study_info.patient_fk;
        return -1;
    }

    //date time
    if (!study_info.study_date.empty()) {
        if (0 != TimeUtil::check_yyyymmdd(study_info.study_date)) {
            MI_IO_LOG(MI_ERROR) << "invalid study info: study date: " << study_info.study_date;
            return -1;
        }
    }

    if (!study_info.study_time.empty()) {
        if (0 != TimeUtil::check_hhmmss(study_info.study_time)) {
            MI_IO_LOG(MI_ERROR) << "invalid study info: study time: " << study_info.study_time;
            return -1;
        }
    }

    //study_id
    if (study_info.study_id.size() > 16) {
        MI_IO_LOG(MI_ERROR) << "invalid study info: study id: " << study_info.study_id;
        return -1;
    }

    //study_uid
    if (study_info.study_uid.size() > UID_LIMIT) {
        MI_IO_LOG(MI_ERROR) << "invalid study info: study uid: " << study_info.study_uid;
        return -1;
    }

    //accession number
    if (study_info.accession_no.size() > 16) {
        MI_IO_LOG(MI_ERROR) << "invalid study info: accession number: " << study_info.accession_no;
        return -1;
    }

    //study description
    if (study_info.study_desc.size() > DESCRIPTION_LIMIT) {
        MI_IO_LOG(MI_ERROR) << "invalid study info: study description: " << study_info.study_desc;
        return -1;
    }

    //series number
    if (study_info.num_series <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid study info: number of series: " << study_info.num_series;
        return -1;
    }

    //instance number
    if (study_info.num_series <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid study info: number of instance: " << study_info.num_instance;
        return -1;
    }

    return 0;
}

int DB::verfiy_series_info(const SeriesInfo& series_info) {
    //verify
    //study_fk
    if (series_info.study_fk <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid study fk: study fk: " << series_info.study_fk;
        return -1;
    }

    //serise_no
    if (series_info.series_no.size() > 16) {
        MI_IO_LOG(MI_ERROR) << "invalid series info: study id: " << series_info.series_no;
        return -1;
    }

    //series_uid
    if (series_info.series_uid.size() > UID_LIMIT) {
        MI_IO_LOG(MI_ERROR) << "invalid series info: series_uid: " << series_info.series_uid;
        return -1;
    }

    //modality
    if (series_info.modality.size() > 16) {
        MI_IO_LOG(MI_ERROR) << "invalid series info: modality: " << series_info.modality;
        return -1;
    }

    //institution
    if (series_info.institution.size() > 16) {
        MI_IO_LOG(MI_ERROR) << "invalid series info: institution: " << series_info.institution;
        return -1;
    }

    //series description
    if (series_info.series_desc.size() > DESCRIPTION_LIMIT) {
        MI_IO_LOG(MI_ERROR) << "invalid series info: series description: " << series_info.series_desc;
        return -1;
    }

    //instance number
    if (series_info.num_instance <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid series info: number of instance: " << series_info.num_instance;
        return -1;
    }

    return 0;
}

int DB::insert_patient(PatientInfo& patient_info) {
    TRY_CONNECT
    
    if (patient_info.md5.empty()) {
        if(0 != get_patient_hash(patient_info, patient_info.md5)) {
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
        int err = this->query(sql.str(), res); 
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            MI_IO_LOG(MI_ERROR) << "db insert patient failed.";
            return -1;
        }
    }

    return 0;    
}

int DB::update_patient(PatientInfo& patient_info) {
    TRY_CONNECT
    
    //verify
    if (patient_info.id <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid patient pk.";
        return -1;
    }

    if (patient_info.md5.empty()) {
        if(0 != get_patient_hash(patient_info, patient_info.md5)) {
            MI_IO_LOG(MI_ERROR) << "calcualte patient md5 failed.";
            return -1;
        }
    }
    std::stringstream sql;
    {
        sql << "UPDATE " << PATIENT_TABLE << " SET "
        << "patient_id=\'" << patient_info.patient_id << "\',\'"
        << "patient_sex=\'" << patient_info.patient_sex << "\',\'"
        << "patient_name=\'" << patient_info.patient_name << "\',\'"
        << "patient_birth_date=\'" << patient_info.patient_birth_date << "\',\'"
        << "WHERE id=" << patient_info.id << ";";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res); 
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            MI_IO_LOG(MI_ERROR) << "db insert patient failed.";
            return -1;
        }
    }

    return 0;    
}


int DB::insert_study(StudyInfo& study_info) {
    TRY_CONNECT

    //verify
    if (0 != verfiy_study_info(study_info)) {
        MI_IO_LOG(MI_ERROR) << "insert study failed: invalid study info.";
        return -1;
    }

    std::string datetime;
    if (!study_info.study_date.empty()) {
        if (study_info.study_time.empty()) {
            datetime = study_info.study_date + "000000";
        } else {
            datetime = study_info.study_date + study_info.study_time;
        }
    }

    std::stringstream sql;
    sql << "INSERT INTO " << STUDY_TABLE << "(patient_fk, study_id, study_uid, ";
    if (datetime.empty()) {
        sql << "study_date_time, accession_no, study_desc, num_series, num_instance)";
    } else {
        sql << "accession_no, study_desc, num_series, num_instance)";
    }
    sql << " VALUES("
    << "\'" << study_info.patient_fk << "\',"
    << "\'" << study_info.study_id << "\',"
    << "\'" << study_info.study_uid << "\',";
    if (!datetime.empty()) {
        sql << "\'" << datetime << "\',";
    }
    sql << "\'" << study_info.accession_no << "\',"
    << "\'" << study_info.study_desc << "\',"
    << "\'" << study_info.num_series << "\',"
    << "\'" << study_info.num_instance << "\'"
    << ");";

    sql::ResultSet* res = nullptr;
    if(0 != this->query(sql.str(), res) ) {
        StructShield<sql::ResultSet> shield(res);
        MI_IO_LOG(MI_ERROR) << "db insert study failed.";
        return -1;
    }
    
    return 0;
}

int DB::update_study(StudyInfo& study_info) {
    TRY_CONNECT

    //verify
    if (0 != verfiy_study_info(study_info)) {
        MI_IO_LOG(MI_ERROR) << "update study failed: invalid study info.";
        return -1;
    }

    std::string datetime;
    if (!study_info.study_date.empty()) {
        if (study_info.study_time.empty()) {
            datetime = study_info.study_date + "000000";
        } else {
            datetime = study_info.study_date + study_info.study_time;
        }
    }

    std::stringstream sql;
    sql << "UPDATE " << STUDY_TABLE << " SET " 
    << "study_uid=\'" << study_info.study_uid << "\'"
    << "study_id=\'" << study_info.study_id << "\'"
    << "study_datetime=\'" << datetime << "\'"
    << "accession_no=\'" << study_info.accession_no << "\'"
    << "study_desc=\'" << study_info.study_desc << "\'"
    << "num_series=" << study_info.num_series
    << "num_instance=" << study_info.num_instance
    << "WHERE id=" << study_info.id;
    
    sql::ResultSet* res = nullptr;
    int err = query(sql.str(), res);
    StructShield<sql::ResultSet> shield(res);
    if (0 != err) {
        MI_IO_LOG(MI_ERROR) << "update study failed: sql failed.";
        return -1;
    }
    return 0;
}

int DB::insert_series(SeriesInfo& series_info) {
    TRY_CONNECT

    //verify
    if (0 != verfiy_series_info(series_info)) {
        MI_IO_LOG(MI_ERROR) << "insert series failed: invalid series info.";
        return -1;
    }

    std::stringstream sql;
    sql << "INSERT INTO study(study_fk, series_uid, series_no, modality, series_desc, institution, num_instance)"
    << " VALUES("
    << "\'" << series_info.study_fk << "\',"
    << "\'" << series_info.series_uid << "\',"
    << "\'" << series_info.series_no << "\',"
    << "\'" << series_info.modality << "\',"
    << "\'" << series_info.series_desc << "\',"
    << "\'" << series_info.institution << "\',"
    << "\'" << series_info.num_instance << "\'"
    << ");";

    sql::ResultSet* res = nullptr;
    if(0 != this->query(sql.str(), res) ) {
        StructShield<sql::ResultSet> shield(res);
        MI_IO_LOG(MI_ERROR) << "db insert series failed.";
        return -1;
    }
    
    return 0;
}

int DB::update_series(SeriesInfo& series_info) {
    TRY_CONNECT

    //verify
    if (0 != verfiy_series_info(series_info)) {
        MI_IO_LOG(MI_ERROR) << "insert series failed: invalid series info.";
        return -1;
    }

    std::stringstream sql;
    sql << "INSERT INTO study(study_fk, series_uid, series_no, modality, series_desc, institution, num_instance)"
    << " VALUES("
    << "\'" << series_info.study_fk << "\',"
    << "\'" << series_info.series_uid << "\',"
    << "\'" << series_info.series_no << "\',"
    << "\'" << series_info.modality << "\',"
    << "\'" << series_info.series_desc << "\',"
    << "\'" << series_info.institution << "\',"
    << "\'" << series_info.num_instance << "\'"
    << ");";

    sql::ResultSet* res = nullptr;
    if(0 != this->query(sql.str(), res) ) {
        StructShield<sql::ResultSet> shield(res);
        MI_IO_LOG(MI_ERROR) << "db insert series failed.";
        return -1;
    }
    
    return 0;
}

int DB::insert_instance(const std::string& user_fk, int64_t series_fk, const std::vector<DcmInstanceInfo>& instance_info) {
    TRY_CONNECT

    //verify
    //series_fk
    if (series_fk <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid instance info: series_fk: " << series_fk;
        return -1;
    }

    //user_fk
    if (user_fk.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid instance info: user_fk: " << user_fk;
        return -1;
    }

    //instance number
    if (instance_info.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid instance info: emtpy.";
        return -1;
    }

    for (auto it = instance_info.begin(); it != instance_info.end(); ++it) {
        const DcmInstanceInfo &info = *it;
        //verfiy
        if (info.sop_class_uid.empty()) {
            MI_IO_LOG(MI_ERROR) << "invalid instance info: emtpy sop class uid.";
            return -1;
        }
        if (info.sop_instance_uid.empty()) {
            MI_IO_LOG(MI_ERROR) << "invalid instance info: emtpy sop instance uid.";
            return -1;
        }
        if (info.file_path.empty()) {
            MI_IO_LOG(MI_ERROR) << "invalid instance info: emtpy file path.";
            return -1;
        }
        if (info.file_size <=0) {
            MI_IO_LOG(MI_ERROR) << "invalid instance info: emtpy file size.";
            return -1;
        }

        std::stringstream sql;
        sql << "INSERT INTO study(series_fk, sop_class_uid, sop_instance_uid, retrieve_user_fk, file_path, file_size)"
        << " VALUES("
        << "\'" << series_fk << "\',"
        << "\'" << info.sop_class_uid << "\',"
        << "\'" << info.sop_instance_uid << "\',"
        << "\'" << user_fk << "\',"
        << "\'" << info.file_path << "\',"
        << "\'" << info.file_size << "\',"
        << ");";

        sql::ResultSet* res = nullptr;
        if(0 != this->query(sql.str(), res) ) {
            StructShield<sql::ResultSet> shield(res);
            MI_IO_LOG(MI_ERROR) << "db insert instnace failed.";
            return -1;
        }
    }

    return 0;
}

int DB::query_patient(const PatientInfo& key, std::vector<PatientInfo>* patient_infos) {
    TRY_CONNECT

    if (!patient_infos) {
        MI_IO_LOG(MI_ERROR) << "patient infos is null.";
        return -1;
    }

    patient_infos->clear();
    std::stringstream sql;
    sql << "SELECT id, patient_id, patient_name, patient_sex, patient_birth_date, md5 FROM patient WHERE ";
    if (key.id > 0) {
       sql << "id=" << key.id << " AND "; 
    }
    if (!key.patient_id.empty()) {
       sql << "patient_id=\'" << key.patient_id << "\' AND "; 
    }
    if (!key.patient_name.empty()) {
       sql << "patient_name=\'" << key.patient_name << "\' AND "; 
    }
    if (!key.patient_sex.empty()) {
       sql << "patient_sex=\'" << key.patient_sex << "\' AND "; 
    }
    if (!key.md5.empty()) {
       sql << "md5=\'" << key.md5 << "\' AND "; 
    }
    sql << "1";

    sql::ResultSet* res = nullptr;
    if(0 != this->query(sql.str(), res) ) {
        StructShield<sql::ResultSet> shield(res);
        MI_IO_LOG(MI_ERROR) << "db query patient failed.";
        return -1;
    } else {
        StructShield<sql::ResultSet> shield(res);
        while (res->next()) {
            patient_infos->push_back(PatientInfo());
            PatientInfo& info = (*patient_infos)[patient_infos->size()-1];
            info.id = res->getInt64("id");
            info.patient_id = res->getString("patient_id");
            info.patient_name = res->getString("patient_name");
            info.patient_sex = res->getString("patient_sex");
            info.patient_birth_date = res->getString("patient_birth_date");
            info.md5 = res->getString("md5");
        }
    }

    return 0;
}

int DB::insert_dcm_series(StudyInfo& study_info, SeriesInfo& series_info, PatientInfo& patient_info, UserInfo& user_info, 
         const std::vector<DcmInstanceInfo>& instance_info) {
    TRY_CONNECT
    MI_IO_LOG(MI_DEBUG) << "insert dcm series: IN";
    //transaction begin
    int err = 0;
    sql::ResultSet* res_begin = nullptr;
    err = this->query("BEGIN;", res_begin);
    StructShield<sql::ResultSet> shield(res_begin);
    if(0 != err) {
        MI_IO_LOG(MI_ERROR) << "db query transaction begin failed.";
        return -1;
    }
    
    MI_IO_LOG(MI_DEBUG) << "insert dcm series: begin";

    try {
        //---------------------------//
        //insert into patient
        //---------------------------//
        if (patient_info.md5.empty()) {
            if (0 != get_patient_hash(patient_info, patient_info.md5)) {
                throw std::exception(std::logic_error("calculate patient md5 failed."));
            }
        }

        //get patient pk
        PatientInfo pkey;
        pkey.md5 = patient_info.md5;
        std::vector<PatientInfo> pres;
        if (0 != query_patient(pkey, &pres)) {
            throw std::exception(std::logic_error("query patient failed."));
        }

        if (pres.size() == 0) {
            if (0 != insert_patient(patient_info)) {
                throw std::exception(std::logic_error("insert patient failed."));
            }
            if (0 != query_patient(pkey, &pres)) {
               throw std::exception(std::logic_error("query patient failed."));
            }
        }
        study_info.patient_fk = pres[0].id;

        MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert patient done";

        //---------------------------//
        //insert into study
        //---------------------------//
        //query study
        if (study_info.study_uid.empty() || study_info.study_uid.size() > UID_LIMIT) {
            throw std::exception(std::logic_error("invalid study uid."));
        }
        
        int64_t study_pk = -1;
        {
            std::string sql_study("SELECT study_uid FROM study WHERE study_uid=\'");
            sql_study += study_info.study_uid;
            sql_study += "\';";
            sql::ResultSet* res_study = nullptr;
            err = query(sql_study, res_study);
            StructShield<sql::ResultSet> shield_res_study(res_study);
            if (0 != err) {
                throw std::exception(std::logic_error("query study failed."));
            }
            if (res_study->next()) {
                study_pk = res_study->getInt64("id");
            }
        }

        if (study_pk <= 0) {
            //insert study
            if(0 != insert_study(study_info)) {
                throw std::exception(std::logic_error("insert study failed."));
            }
        } else {
            //update study
            study_info.id = study_pk;
            if (0 != update_study(study_info) ) {
                throw std::exception(std::logic_error("update study failed."));
            }            
        }

        MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert study done";

        //---------------------------//
        //insert into series
        //---------------------------//
        //query series
        series_info.study_fk = study_pk;
        if (series_info.series_uid.empty() || series_info.series_uid.size() > UID_LIMIT) {
            throw std::exception(std::logic_error("invalid series uid."));
        }
        
        int64_t series_pk = -1;
        {
            std::string sql_series("SELECT series_uid FROM series WHERE series_uid=\'");
            sql_series += series_info.series_uid;
            sql_series += "\';";
            sql::ResultSet* res_series = nullptr;
            err = query(sql_series, res_series);
            StructShield<sql::ResultSet> shield_res_series(res_series);
            if (0 != err) {
                throw std::exception(std::logic_error("query series failed."));
            }
            if (res_series->next()) {
                series_pk = res_series->getInt64("id");
            }
        }

        if (series_pk <= 0) {
            //insert series
            if(0 != insert_series(series_info)) {
                throw std::exception(std::logic_error("insert series failed."));
            }
        } else {
            //update series
            series_info.id = series_pk;
            if (0 != update_series(series_info) ) {
                throw std::exception(std::logic_error("update series failed."));
            }            
        }

        MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert series done";

        //---------------------------//
        //insert into series
        //---------------------------//
        if (user_info.id.empty()) {
            throw std::exception(std::logic_error("invalid user fk."));
        }
        if (0 != insert_instance(user_info.id, series_pk, instance_info)) {
            throw std::exception(std::logic_error("insert instance failed."));
        }

        MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert instance done";
        
    } catch (const std::exception& e) {
        //transaction rollback
        sql::ResultSet* res_rollback = nullptr;
        err = this->query("ROLLBACK;", res_rollback);
        StructShield<sql::ResultSet> shield(res_rollback);
        if(0 != err) {
            MI_IO_LOG(MI_ERROR) << "db query transaction rollback failed.";
        }
        return -1;
    }

    //transaction commit
    sql::ResultSet* res_commit = nullptr;
    err = this->query("COMMIT;", res_commit);
    StructShield<sql::ResultSet> shield2(res_commit);
    if(0 != err) {
        MI_IO_LOG(MI_ERROR) << "db query transaction commit failed.";
    }

    MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert done";

    return 0;
}

int DB::query_user(const UserInfo& key, std::vector<UserInfo>* user_infos) {
    TRY_CONNECT

    if (!user_infos) {
        MI_IO_LOG(MI_ERROR) << "query user failed: null user info input.";
        return -1;
    }

    std::stringstream sql;
    sql << "SELECT id,name,role_fk FROM " << USER_TABLE << " WHERE ";
    if (!key.id.empty()) {
        sql << "id=\'" << key.id << "\'";
    } else {
        if (!key.name.empty()) {
            sql << "name=\'" << key.name << "\' AND ";
        }
        if (key.role_fk > 0) {
            sql << "role_fk=" << key.role_fk << "\' AND ";
        }
        sql << "1";
    }

    sql::ResultSet* res = nullptr;
    MI_IO_LOG(MI_DEBUG) << "SQL: " << sql.str();
    int err = query(sql.str(), res);
    if (0 != err) {
        MI_IO_LOG(MI_ERROR) << "query user failed: sql error";
        return -1;
    } else {
        user_infos->clear();
        while(res->next()) {
            user_infos->push_back(UserInfo(
                res->getString("id").asStdString(),
                res->getInt("role_fk"),
                res->getString("name").asStdString()
            ));
        }
    }

    return 0;
}

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
//     if(0 != query_dcm_item(item.series_id, in_db)) {
//         MI_IO_LOG(MI_ERROR) << "query failed when insert item.";
//         return -1;
//     }

//     //delete if exit
//     if (in_db) {
//         if(0 != delete_dcm_item(item.series_id)) {
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
//     if(0 != query_dcm_item(series_id, in_db)) {
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