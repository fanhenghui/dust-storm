#include "mi_db.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "cppconn/connection.h"
#include "mysql_connection.h"
#include "mi_io_logger.h"
#include "mi_md5.h"
#include "util/mi_memory_shield.h"
#include "util/mi_time_util.h"
#include "util/mi_file_util.h"

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
const static std::string ANNOTATION_TABLE = "annotation";

const static int UID_LIMIT = 64;
const static int DESCRIPTION_LIMIT = 64;

#define TRY_CONNECT \
if (!this->try_connect()) {\
    MI_IO_LOG(MI_ERROR) << "db connection invalid.";\
    return -1;\
}

#define THROW_SQL_EXCEPTION throw std::exception(std::logic_error("sql error"));

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

    try {
        std::stringstream sql;
        if (patient_info.patient_birth_date.empty()) {
            sql << "INSERT INTO " << PATIENT_TABLE <<"(patient_id,patient_name,patient_sex,md5)" << " VALUES (\'";
        } else {
            sql << "INSERT INTO " << PATIENT_TABLE <<"(patient_id,patient_name,patient_sex,patient_birth_date,md5)" << " VALUES (\'";
        }
        
        sql << patient_info.patient_id << "\',\'" 
        << patient_info.patient_name << "\',\'" 
        << patient_info.patient_sex << "\',\'";
        if (!patient_info.patient_birth_date.empty()) {
            sql << patient_info.patient_birth_date << "\',\'";
        }
        sql << patient_info.md5 << "\')";

        MI_IO_LOG(MI_DEBUG) << "SQL: " << sql.str(); 

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res); 
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            THROW_SQL_EXCEPTION
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db insert patient failed: " << e.what();
        return -1;
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

    try {
        std::stringstream sql;
        sql << "UPDATE " << PATIENT_TABLE << " SET "
        << "patient_id=\'" << patient_info.patient_id << "\',"
        << "patient_sex=\'" << patient_info.patient_sex << "\',"
        << "patient_name=\'" << patient_info.patient_name << "\',"
        << "patient_birth_date=\'" << patient_info.patient_birth_date << "\' "
        << "WHERE id=" << patient_info.id << ";";
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res); 
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            THROW_SQL_EXCEPTION
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db update patient failed: " << e.what();
        return -1;
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

    try {
        std::stringstream sql;
        sql << "INSERT INTO " << STUDY_TABLE << "(patient_fk, study_id, study_uid, ";
        if (!datetime.empty()) {
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

        MI_IO_LOG(MI_DEBUG) << "SQL: " << sql.str();

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            THROW_SQL_EXCEPTION
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db insert study failed: " << e.what();
        return -1;
    }
    
    return 0;
}

int DB::update_study(StudyInfo& study_info) {
    TRY_CONNECT

    if (study_info.id <= 0) {
        MI_IO_LOG(MI_ERROR) << "update study failed: invalid study pk.";
        return -1;
    }

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

    try {
        std::stringstream sql;
        sql << "UPDATE " << STUDY_TABLE << " SET " 
        << "study_uid=\'" << study_info.study_uid << "\',"
        << "study_id=\'" << study_info.study_id << "\',";
        if (!datetime.empty()) {
            sql << "study_date_time=\'" << datetime << "\',";
        }
        sql << "accession_no=\'" << study_info.accession_no << "\',"
        << "study_desc=\'" << study_info.study_desc << "\',"
        << "num_series=" << study_info.num_series << ","
        << "num_instance=" << study_info.num_instance << " "
        << "WHERE id=" << study_info.id;

        sql::ResultSet* res = nullptr;
        int err = query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            THROW_SQL_EXCEPTION
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db update study failed: " << e.what();
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

    try { 
        std::stringstream sql;
        sql << "INSERT INTO " << SERIES_TABLE
        << "(study_fk, series_uid, series_no, modality, series_desc, institution, num_instance) VALUES("
        << "\'" << series_info.study_fk << "\',"
        << "\'" << series_info.series_uid << "\',"
        << "\'" << series_info.series_no << "\',"
        << "\'" << series_info.modality << "\',"
        << "\'" << series_info.series_desc << "\',"
        << "\'" << series_info.institution << "\',"
        << "\'" << series_info.num_instance << "\'"
        << ");";

        MI_IO_LOG(MI_DEBUG) << "SQL: " << sql.str(); 

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            THROW_SQL_EXCEPTION
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db insert series failed: " << e.what();
        return -1;
    }
    
    return 0;
}

int DB::update_series(SeriesInfo& series_info) {
    TRY_CONNECT

    if (series_info.id <= 0) {
        MI_IO_LOG(MI_ERROR) << "update series failed: invalid series pk.";
        return -1;
    }

    //verify
    if (0 != verfiy_series_info(series_info)) {
        MI_IO_LOG(MI_ERROR) << "insert series failed: invalid series info.";
        return -1;
    }

    try { 
        std::stringstream sql;
        sql << "UPDATE " << SERIES_TABLE << " SET " 
        << "study_fk=" << series_info.study_fk << ", "
        << "series_uid=\'" << series_info.series_uid << "\', " 
        << "series_no=\'" << series_info.series_no << "\', " 
        << "modality=\'" << series_info.modality << "\', " 
        << "series_desc=\'" << series_info.series_desc << "\', " 
        << "institution=\'" << series_info.institution << "\', " 
        << "num_instance=" << series_info.num_instance 
        << " WHERE id=" << series_info.id << ";";

        MI_IO_LOG(MI_DEBUG) << "SQL: " << sql.str();

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            THROW_SQL_EXCEPTION
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db update series failed: " << e.what();
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

    try {
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
            sql << "INSERT INTO " << INSTANCE_TABLE
            << "(series_fk, sop_class_uid, sop_instance_uid, retrieve_user_fk, file_path, file_size) VALUES("
            << "\'" << series_fk << "\',"
            << "\'" << info.sop_class_uid << "\',"
            << "\'" << info.sop_instance_uid << "\',"
            << "\'" << user_fk << "\',"
            << "\'" << info.file_path << "\',"
            << "\'" << info.file_size << "\'"
            << ");";

            sql::ResultSet* res = nullptr;
            int err = this->query(sql.str(), res);
            StructShield<sql::ResultSet> shield(res);
            if(0 != err) {
                THROW_SQL_EXCEPTION
            }
        }
    } catch(const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db insert instance failed: " << e.what();
        return -1;
    }
    return 0;
}

int DB::insert_dcm_series(StudyInfo& study_info, SeriesInfo& series_info, PatientInfo& patient_info, UserInfo& user_info, 
         const std::vector<DcmInstanceInfo>& instance_info) { 

    TRY_CONNECT

    MI_IO_LOG(MI_DEBUG) << "insert dcm series: IN";
    //transaction begin
    int err = 0;
    MI_IO_LOG(MI_DEBUG) << "insert dcm series: begin";

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("insert_dcm_series");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        //---------------------------//
        //delete old series
        //---------------------------//
        if (series_info.series_uid.size() > UID_LIMIT) {
            throw std::exception(std::logic_error("invalid series uid."));
        }

        std::stringstream sql_select_series;
        sql_select_series << "SELECT id FROM " << SERIES_TABLE
        << " WHERE series_uid=\'" << series_info.series_uid << "\';";

        sql::ResultSet* res_select_series = nullptr;
        err = query(sql_select_series.str(), res_select_series);
        StructShield<sql::ResultSet> shield_select_series(res_select_series);
        if (0 != err) {
            throw std::exception(std::logic_error("query study failed."));
        } else if (res_select_series->next()) {
            if (0!= delete_dcm_series(res_select_series->getInt64("id"), false)) {
                throw std::exception(std::logic_error("delete series failed."));    
            }
        }

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
            //insert new patient
            if (0 != insert_patient(patient_info)) {
                throw std::exception(std::logic_error("insert patient failed."));
            }
            if (0 != query_patient(pkey, &pres)) {
               throw std::exception(std::logic_error("query patient failed."));
            }
        } else {
            //update patient info
            if (0 != update_patient(pres[0])) {
                throw std::exception(std::logic_error("update patient failed."));
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
            std::string sql_study("SELECT id FROM study WHERE study_uid=\'");
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

            std::string sql_study("SELECT id FROM study WHERE study_uid=\'");
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
            std::string sql_series("SELECT id FROM series WHERE series_uid=\'");
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

        if (series_pk > 0) {
            //delete old series
            if (0!= delete_dcm_series(series_pk, false)) {
                throw std::exception(std::logic_error("delete series failed."));    
            }
            series_pk = -1;
        }
        
        if(0 != insert_series(series_info)) {
            throw std::exception(std::logic_error("insert series failed."));
        } else {
            //get series pk
            std::string sql_series("SELECT id FROM series WHERE series_uid=\'");
            sql_series += series_info.series_uid;
            sql_series += "\';";
            sql::ResultSet* res_series = nullptr;
            err = query(sql_series, res_series);
            StructShield<sql::ResultSet> shield_res_series(res_series);
            if (0 != err) {
                throw std::exception(std::logic_error("query series failed after insert new series."));
            }
            if (res_series->next()) {
                series_pk = res_series->getInt64("id");
            }
        }

        MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert series done";

        //---------------------------//
        //insert into instance
        //---------------------------//
        if (user_info.id.empty()) {
            throw std::exception(std::logic_error("invalid user fk."));
        }
        if (0 != insert_instance(user_info.id, series_pk, instance_info)) {
            throw std::exception(std::logic_error("insert instance failed."));
        }

        MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert instance done";
        
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "db insert dcm series failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }

    _connection->commit();
    MI_IO_LOG(MI_DEBUG) << "insert dcm series: insert done";

    return 0;
}

int DB::delete_dcm_series(int series_fk, bool transcation) {
    
    sql::Savepoint* save_point = nullptr;
    if (transcation) {
        _connection->setAutoCommit(false);
        save_point = _connection->setSavepoint("delete_dcm_series.");
    }
    StructShield<sql::Savepoint> shield0(save_point);
    
    try {
        int err = 0;
        
        //-------------------------------------------------------------//
        //delete instance file
        //-------------------------------------------------------------//
        std::stringstream sql;
        sql << "SELECT file_path FROM instance WHERE series_fk=" << series_fk << ";";
        sql::ResultSet* res_select_instance = nullptr;
        StructShield<sql::ResultSet> shield(res_select_instance);
        err = this->query(sql.str(), res_select_instance);
        if (err != 0) {
            throw std::exception(std::logic_error("query instance failed."));
        }

        while(res_select_instance->next()) {
            std::string file_path = res_select_instance->getString("file_path");
            if (0 != FileUtil::remove_file(file_path)) {
                MI_IO_LOG(MI_WARNING) << "remove instance file: "<< file_path << " failed.";
            }
        }

        //-------------------------------------------------------------//
        //delete instance row
        //-------------------------------------------------------------//
        sql.str("");
        sql << "DELETE FROM instance WHERE series_fk=" << series_fk << ";";
        sql::ResultSet* res_delete_instance = nullptr;
        StructShield<sql::ResultSet> shield2(res_delete_instance);
        err = this->query(sql.str(), res_delete_instance);
        if (err != 0) {
            throw std::exception(std::logic_error("delete instance row failed."));
        }

        //-------------------------------------------------------------//
        //delete series
        //-------------------------------------------------------------//
        //query study pk
        sql.str("");
        sql << "SELECT study_fk FROM series WHERE id=" << series_fk << ";";
        sql::ResultSet* res_select_series = nullptr;
        StructShield<sql::ResultSet> shield3(res_select_series);
        err = this->query(sql.str(), res_select_series);
        if (err != 0) {
            throw std::exception(std::logic_error("select series row failed."));
        }

        int64_t study_fk = -1;
        if(res_select_series->next()) {
            study_fk = res_select_series->getInt64("study_fk"); 
        }
        if (study_fk <= 0) {
            throw std::exception(std::logic_error("invalid study fk in this series."));
        }   

        sql.str("");
        sql << "DELETE FROM series WHERE id=" << series_fk << ";";
        sql::ResultSet* res_delete_series = nullptr;
        StructShield<sql::ResultSet> shield4(res_delete_series);
        err = this->query(sql.str(), res_delete_series);
        if (err != 0) {
            throw std::exception(std::logic_error("delete series row failed."));
        }

        //-------------------------------------------------------------//
        //delete/update study when study has just one series
        //-------------------------------------------------------------//
        sql.str("");
        sql << "SELECT patient_fk, num_series FROM study WHERE id=" << study_fk << ";";
        sql::ResultSet* res_select_study = nullptr;
        StructShield<sql::ResultSet> shield5(res_select_study);
        err = this->query(sql.str(), res_select_study);
        if (err != 0) {
            throw std::exception(std::logic_error("select study row failed."));
        }

        int num_series = -1;
        int64_t patient_fk = -1;
        if (res_select_study->next()) {
            num_series = res_select_study->getInt("num_series");
            patient_fk = res_select_study->getInt64("patient_fk");
        } else {
            throw std::exception(std::logic_error("select study row failed: null study."));
        }

        if (1 == num_series) {
            //delete study row
            sql.str("");
            sql << "DELETE FROM study WHERE id=" << study_fk << ";";
            sql::ResultSet* res_delete_study = nullptr;
            StructShield<sql::ResultSet> shield(res_delete_study);
            err = this->query(sql.str(), res_delete_study);
            if (err != 0) {
                throw std::exception(std::logic_error("delete study row failed."));
            }
        } else {
            //update study num_series
            sql.str("");
            sql << "UPDATE " << STUDY_TABLE <<  " SET num_series="
            << (num_series-1)
            << " WHERE id=" << study_fk << ";";
            sql::ResultSet* res_update_study = nullptr;
            StructShield<sql::ResultSet> shield(res_update_study);
            err = this->query(sql.str(), res_update_study);
            if (err != 0) {
                throw std::exception(std::logic_error("update study row failed."));
            }
        }

        //-------------------------------------------------------------//
        //delete patient when patient has just one study(has one series)
        //-------------------------------------------------------------//
        if (num_series == 1) {
            sql.str("");
            sql << "SELECT id FROM study WHERE patient_fk=" << patient_fk << ";";
            sql::ResultSet* res_select_study2 = nullptr;
            StructShield<sql::ResultSet> shield(res_select_study2);
            err = this->query(sql.str(), res_select_study2);
            if (err != 0) {
                throw std::exception(std::logic_error("delete study row failed."));
            }

            if (!res_select_study2->next()) {
                //delete patient
                sql.str("");
                sql << "DELETE FROM patient WHERE id=" << patient_fk << ";";
                sql::ResultSet* res_delete_patient = nullptr;
                StructShield<sql::ResultSet> shield(res_delete_patient);
                err = this->query(sql.str(), res_delete_patient);
                if (err != 0) {
                    throw std::exception(std::logic_error("delete patient row failed."));
                }
            }
        }
    }
    catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "delete series(pk): " << series_fk << " failed:" << e.what();
        if (transcation) {
            _connection->rollback(save_point);
        }
        return -1;
    }

    if (transcation) {
        _connection->commit();
    }

    return 0;
}

int DB::verify_evaluation_info(const EvaluationInfo& info) {
    if (info.series_fk < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid evaulation series id.";    
        return -1;    
    }

    if (info.eva_type < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid evaulation type id.";
        return -1;    
    }

    if (info.eva_version.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid evaulation version.";
        return -1;    
    }

    if (info.eva_file_path.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid evaulation file path.";
        return -1;    
    }

    if (info.eva_file_size <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid evaulation file size.";
        return -1;    
    }

    return 0;
}

int DB::verify_annotation_info(const AnnotationInfo& info) {
    if (info.series_fk < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid annotation series id.";    
        return -1;    
    }

    if (info.user_id < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid user id.";    
        return -1;    
    }

    if (info.anno_type < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid annotation type id.";
        return -1;    
    }

    if (info.anno_file_path.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid annotation file path.";
        return -1;    
    }

    if (info.anno_file_size <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid annotation file size.";
        return -1;    
    }

    return 0;    
}

int DB::verify_preprocess_info(const PreprocessInfo& info) {
    if (info.series_fk < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid preprocess series id.";    
        return -1;    
    }

    if (info.prep_type < 1) {
        MI_IO_LOG(MI_ERROR) << "invalid preprocess type id.";
        return -1;    
    }

    if (info.prep_version.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid preprocess version.";
        return -1;    
    }

    if (info.prep_file_path.empty()) {
        MI_IO_LOG(MI_ERROR) << "invalid preprocess file path.";
        return -1;    
    }

    if (info.prep_file_size <= 0) {
        MI_IO_LOG(MI_ERROR) << "invalid preprocess file size.";
        return -1;    
    }

    return 0;    
}

int DB::insert_evaluation(const EvaluationInfo& eva_info) {
    if (0 != verify_evaluation_info(eva_info)) {
        MI_IO_LOG(MI_ERROR) << "insert evaluation failed: invalid evaluation info.";
        return -1;
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("insert_evaluation");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "INSERT INTO " << EVALUATION_TABLE 
        << "(series_fk, eva_type, eva_version, eva_file_path, eva_file_size) VALUES("
        << "\'" << eva_info.series_fk << "\',"
        << "\'" << eva_info.eva_type << "\',"
        << "\'" << eva_info.eva_version << "\',"
        << "\'" << eva_info.eva_file_path << "\',"
        << "\'" << eva_info.eva_file_size << "\');";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "insert evaluation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();
    
    return 0;
}

int DB::insert_annotation(const AnnotationInfo& anno_info) {
    if (0 != verify_annotation_info(anno_info)) {
        MI_IO_LOG(MI_ERROR) << "insert evaluation failed: invalid annotation info.";
        return -1;
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("insert_annotation");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "INSERT INTO " << ANNOTATION_TABLE 
        << "(series_fk, user_id, anno_type, anno_desc, eva_file_path, eva_file_size) VALUES("
        << "\'" << anno_info.series_fk << "\',"
        << "\'" << anno_info.user_id << "\',"
        << "\'" << anno_info.anno_type << "\',"
        << "\'" << anno_info.anno_desc << "\',"
        << "\'" << anno_info.anno_file_path << "\',"
        << "\'" << anno_info.anno_file_size << "\');";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "insert annotation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();
    
    return 0;
}

int DB::insert_preprocess(const PreprocessInfo& prep_info) {
    if (0 != verify_preprocess_info(prep_info)) {
        MI_IO_LOG(MI_ERROR) << "insert preprocess failed: invalid preprocess info.";
        return -1;
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("insert_preprocess");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "INSERT INTO " << PREPROCESS_TABLE 
        << "(series_fk, prep_type, prep_version, prep_file_path, prep_file_size) VALUES("
        << "\'" << prep_info.series_fk << "\',"
        << "\'" << prep_info.prep_type << "\',"
        << "\'" << prep_info.prep_version << "\',"
        << "\'" << prep_info.prep_file_path << "\',"
        << "\'" << prep_info.prep_file_size << "\');";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "insert annotation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();
    
    return 0;
}

int DB::update_evaluation(const EvaluationInfo& eva_info) {
    if (eva_info.id < 1) {
        MI_IO_LOG(MI_ERROR) << "update evaluation failed: invalid evaluation id.";
        return -1;
    }

    if (0 != verify_evaluation_info(eva_info)) {
        MI_IO_LOG(MI_ERROR) << "update evaluation failed: invalid evaluation info.";
        return -1;
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("update_evaluation");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "UPDATE " << EVALUATION_TABLE << " SET "
        << "series_fk=\'" << eva_info.series_fk << "\',"
        << "eva_type=\'" << eva_info.eva_type << "\',"
        << "eva_version=\'" << eva_info.eva_version << "\',"
        << "eva_file_path=\'" << eva_info.eva_file_path << "\',"
        << "eva_file_size=\'" << eva_info.eva_file_size << "\' "
        << "WHERE id=" << eva_info.id << ";";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "update evaluation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();
    
    return 0;
}

int DB::update_annotation(const AnnotationInfo& anno_info) {
    if (anno_info.id < 1) {
        MI_IO_LOG(MI_ERROR) << "update annotation failed: invalid annotation id.";
        return -1;
    }

    if (0 != verify_annotation_info(anno_info)) {
        MI_IO_LOG(MI_ERROR) << "update annotation failed: invalid annotation info.";
        return -1;
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("update_annotation");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "UPDATE " << ANNOTATION_TABLE << " SET "
        << "series_fk=\'" << anno_info.series_fk << "\',"
        << "anno_type=\'" << anno_info.anno_type << "\',"
        << "user_id=\'" << anno_info.user_id << "\',"
        << "anno_desc=\'" << anno_info.anno_desc << "\',"
        << "anno_file_path=\'" << anno_info.anno_file_path << "\',"
        << "anno_file_size=\'" << anno_info.anno_file_size << "\' "
        << "WHERE id=" << anno_info.id << ";";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "update annotation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();
    
    return 0;
}

int DB::update_preprocess(const PreprocessInfo& prep_info) {
    if (prep_info.id < 1) {
        MI_IO_LOG(MI_ERROR) << "update preprocess failed: invalid preprocess id.";
        return -1;
    }

    if (0 != verify_preprocess_info(prep_info)) {
        MI_IO_LOG(MI_ERROR) << "update preprocess failed: invalid preprocess info.";
        return -1;
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("update_preprocess");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "UPDATE " << PREPROCESS_TABLE << " SET "
        << "series_fk=\'" << prep_info.series_fk << "\',"
        << "prep_type=\'" << prep_info.prep_type << "\',"
        << "prep_version=\'" << prep_info.prep_version << "\',"
        << "prep_file_path=\'" << prep_info.prep_file_path << "\',"
        << "prep_file_size=\'" << prep_info.prep_file_size << "\' "
        << "WHERE id=" << prep_info.id << ";";

        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "update preprocess failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();
    
    return 0;
}

int DB::delete_evaluation(int eva_id) {
    if (eva_id < 1) {
        MI_IO_LOG(MI_ERROR) << "delete evaluation failed: invalid eva_id.";
        return -1;    
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("delete_evaluation");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "DELETE FROM " << EVALUATION_TABLE 
        << " WHERE id=" << eva_id << ";";
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);

        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "delete evaluation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();

    return 0;
}

int DB::delete_annotation(int anno_id) {
    if (anno_id < 1) {
        MI_IO_LOG(MI_ERROR) << "delete annotation failed: invalid anno_id.";
        return -1;    
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("delete_annotationn");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "DELETE FROM " << ANNOTATION_TABLE 
        << " WHERE id=" << anno_id << ";";
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);

        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "delete annotation failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();

    return 0;
}

int DB::delete_preprocess(int prep_id) {
    if (prep_id < 1) {
        MI_IO_LOG(MI_ERROR) << "delete preprocess failed: invalid anno_id.";
        return -1;    
    }

    TRY_CONNECT

    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("delete_preprocess");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        std::stringstream sql;
        sql << "DELETE FROM " << PREPROCESS_TABLE 
        << " WHERE id=" << prep_id << ";";
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);

        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));
        }
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "delete preprocess failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }
    _connection->commit();

    return 0;
}

int DB::query_user(const UserInfo& key, std::vector<UserInfo>* user_infos) {
    if (!user_infos) {
        MI_IO_LOG(MI_ERROR) << "query user failed: user infos is null.";
        return -1;
    }

    TRY_CONNECT

    try {
        std::stringstream sql;
        sql << "SELECT id , role_fk, name FROM " << USER_TABLE << " WHERE ";
        if (!key.id.empty()) {
            sql << "id=\'" << key.id << "\' AND ";
        }
        if (key.role_fk > 1) {
            sql << "role_fk=" << key.role_fk << " AND ";
        }
        if (!key.name.empty()) {
            sql << "name=\'" << key.name << "\' AND ";
        }
        sql << "1;";
        
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));    
        }
        user_infos->clear();
        while(res->next()) {
            user_infos->push_back(UserInfo(
                res->getString("id"),
                res->getInt64("role_fk"),
                res->getString("name")
            ));
        }
        
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "query user failed: " << e.what();
        return -1;
    }

    return 0;
}

int DB::query_evaluation(const EvaluationInfo& key, std::vector<EvaluationInfo>* eva_infos) {
    if (!eva_infos) {
        MI_IO_LOG(MI_ERROR) << "query evaluation failed: evaluation infos is null.";
        return -1;
    }

    TRY_CONNECT

    try {
        std::stringstream sql;
        sql << "SELECT id , series_fk, eva_type, eva_version, eva_file_path, eva_file_size FROM " 
        << EVALUATION_TABLE <<  " WHERE ";
        if (key.id > 1) {
            sql << "id=" << key.id << " AND ";
        }
        if (key.series_fk > 1) {
            sql << "series_fk=" << key.series_fk << " AND ";
        }
        if (key.eva_type > 1) {
            sql << "eva_type=" << key.eva_type << " AND ";
        }
        if (!key.eva_version.empty()) {
            sql << "eva_version=\'" << key.eva_version << "\' AND ";
        }
        sql << "1;";
        
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));    
        }
        eva_infos->clear();
        while(res->next()) {
            eva_infos->push_back(EvaluationInfo(
                res->getInt64("id"),
                res->getInt64("series_fk"),
                res->getInt("eva_type"),
                res->getString("eva_version"),
                res->getString("eva_file_path"),
                res->getInt64("eva_file_size")
            ));
        }
        
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "query evaluation failed: " << e.what();
        return -1;
    }

    return 0;
}

int DB::query_annotation(const AnnotationInfo& key, std::vector<AnnotationInfo>* anno_infos) {
    if (!anno_infos) {
        MI_IO_LOG(MI_ERROR) << "query annotation failed: annotation infos is null.";
        return -1;
    }

    TRY_CONNECT

    try {
        std::stringstream sql;
        sql << "SELECT id , series_fk, anno_type, user_id, anno_desc, anno_file_path, anno_file_size FROM " 
        << ANNOTATION_TABLE <<  " WHERE ";
        if (key.id > 1) {
            sql << "id=" << key.id << " AND ";
        }
        if (key.series_fk > 1) {
            sql << "series_fk=" << key.series_fk << " AND ";
        }
        if (key.anno_type > 1) {
            sql << "anno_type=" << key.anno_type << " AND ";
        }
        if (key.user_id > 1) {
            sql << "user_id=\'" << key.user_id << "\' AND ";
        }
        sql << "1;";
        
        sql::ResultSet* res = nullptr;
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if (0 != err) {
            throw std::exception(std::logic_error("sql error"));    
        }
        anno_infos->clear();
        while(res->next()) {
            anno_infos->push_back(AnnotationInfo(
                res->getInt64("id"),
                res->getInt64("series_fk"),
                res->getInt64("user_id"),
                res->getInt("eva_type"),
                res->getString("eva_version"),
                res->getString("eva_file_path"),
                res->getInt64("eva_file_size")
            ));
        }
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "query annotation failed: " << e.what();
        return -1;
    }

    return 0;
}

int DB::query_patient(const PatientInfo& key, std::vector<PatientInfo>* patient_infos) {
    if (!patient_infos) {
        MI_IO_LOG(MI_ERROR) << "patient infos is null.";
        return -1;
    }
    
    TRY_CONNECT

    try {
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
        int err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);
        if(0 != err) {
            THROW_SQL_EXCEPTION
        }

        patient_infos->clear();
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
    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "query patient failed: " << e.what();
        return -1;
    }

    return 0;
}

bool DB::patient_key_valid(const PatientInfo& key) {
    return ( key.id > 1 || !key.patient_id.empty() ||
             !key.patient_name.empty() || !key.patient_sex.empty() || 
             !key.patient_birth_date.empty() || !key.md5.empty()); 
}

bool DB::study_key_valid(const StudyInfo& key) {
    return ( key.id > 1 || key.patient_fk > 1 || 
             !key.study_id.empty() || !key.study_uid.empty() ||
             !key.study_date.empty() || !key.study_time.empty() || 
             !key.accession_no.empty());
}

int DB::query_study(int64_t patient_fk, const StudyInfo& study_key, std::vector<StudyInfo>* study_infos) {
    std::stringstream sql;
    if (!study_key_valid(study_key)) {
        sql << "SELECT id, study_id, study_uid, study_date_time, accession_no, study_desc, num_series, num_instance FROM "
        << STUDY_TABLE << " WHERE patient_fk=" << patient_fk << ";";
    } else {
        sql << "SELECT id, study_id, study_uid, study_date_time, accession_no, study_desc, num_series, num_instance FROM "
        << STUDY_TABLE << " WHERE patient_fk=" << patient_fk << " AND ";
        //Note 这里没有 query (study_pk, num_series, num_instance) 的场景
        if (!study_key.study_uid.empty()) {
            sql << "study_uid=\'" << study_key.study_uid << "\' AND ";
        }
        if (!study_key.study_id.empty()) {
            sql << "study_id=\'" << study_key.study_id << "\' AND ";
        }
        if (!study_key.accession_no.empty()) {
            sql << "accession_no=\'" << study_key.accession_no << "\' AND ";
        }
        sql << "1;";
    }
    
    sql::ResultSet* res = nullptr;
    int err = this->query(sql.str(), res);
    if (0 != err) {
        MI_IO_LOG(MI_ERROR) << "query study failed: sql error";
        return -1;
    }

    //study_infos->clear();
    while(res->next()) {
        study_infos->push_back(StudyInfo());
        StudyInfo& info = (*study_infos)[study_infos->size()-1];
        info.patient_fk = patient_fk;
        info.study_id = res->getString("study_id");
        info.study_uid = res->getString("study_uid");
        std::string datetime = res->getString("study_date_time");
        if (!datetime.empty() && datetime.size() == 14) {
            info.study_date = datetime.substr(0,8);
            info.study_time = datetime.substr(8,6);
        }
        info.study_desc = res->getString("study_desc");
        info.num_series = res->getInt("num_series");
        info.num_instance = res->getInt("num_instance");
    }

    return 0;
}

int DB::query_study(const PatientInfo& patient_key, const StudyInfo& study_key, 
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos) {
    try {
        TRY_CONNECT

        patient_infos->clear();
        study_infos->clear();

        int err = 0;
        const bool pkey_valid = patient_key_valid(patient_key);
        const bool skey_valid = study_key_valid(study_key);
        if (pkey_valid && skey_valid) {
            //query patient , then query study
            std::vector<PatientInfo> inter_patient_infos;
            if (patient_key.id > 1) {
                inter_patient_infos.push_back(PatientInfo());
                inter_patient_infos[inter_patient_infos.size()-1].id = patient_key.id;
            } else {
                if (0 != this->query_patient(patient_key, &inter_patient_infos)) {
                    throw std::exception(std::logic_error("query patient failed."));
                }
                if (inter_patient_infos.empty()) {
                    return 0;
                }
                for (auto it = inter_patient_infos.begin(); it != inter_patient_infos.end(); ++it) {
                    PatientInfo& patient_info = *it;
                    size_t pre_study_count = study_infos->size();
                    if (0 != this->query_study(patient_info.id, study_key, study_infos)) {
                        throw std::exception(std::logic_error("query study by patient fk failed."));
                    }
                    if (study_infos->size() > pre_study_count) {
                        for (size_t i = 0; i < study_infos->size() - pre_study_count; ++i) {
                            patient_infos->push_back(patient_info);
                        }
                    }
                }
            }
        } else if (!pkey_valid && skey_valid) {
            //query study , then query patient
        } else if (pkey_valid && !skey_valid) {
            //query patient , then query study
        } else {
            //query all study , then query patient
        }

        
        std::vector<PatientInfo> inter_patient_infos;
        if (patient_key.id > 1) {
            inter_patient_infos.push_back(PatientInfo());
            inter_patient_infos[inter_patient_infos.size()-1].id = patient_key.id;
        } else {
            
        }
        
        // if (patient_key_valid(patient_key)) {
        //     if (patient_key)
        //     //get patient fk
        //     err = this->query_patient(patient_key, &inter_patient_infos);
        //     if (-1 == err) {
        //         throw std::exception(std::logic_error("query patient failed."));
        //     }
        // }

        // if (inter_patient_infos.empty()) {
        //     return 0;
        // }

        // for (auto it = )

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "query study failed: " << e.what();
        return -1;
    }

    return 0;
}

int DB::query_series(const PatientInfo& patient_key, const StudyInfo& study_key, const SeriesInfo& series_key, 
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos, std::vector<StudyInfo>* series_infos) {
    
    return 0;
}

int DB::query_instance(int series_fk, std::vector<DcmInstanceInfo>* instance_infos) {
    return 0;
}

MED_IMG_END_NAMESPACE