#recreate medical image database (med_img_db)
mysql -u root -p$1 -e"
drop database if exists med_img_db;
create database med_img_db;
show databases;
use med_img_db;

CREATE TABLE role(id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(64) NOT NULL);

CREATE INDEX name ON role(name(64));

CREATE TABLE user(
    id VARCHAR(32) NOT NULL PRIMARY KEY, 
    name VARCHAR(64) NOT NULL,
    role_fk INT NOT NULL, 
    online_token VARCHAR(32), 
    password VARCHAR(32) NOT NULL, 
    CONSTRAINT role_fk FOREIGN KEY (role_fk) REFERENCES role(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE INDEX name ON user(name(64));

CREATE TABLE patient(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(64),
    patient_name VARCHAR(64),
    patient_sex VARCHAR(16),
    patient_birth_date TIMESTAMP,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX patient_id ON patient(patient_id(64));
CREATE INDEX patient_name ON patient(patient_name(64));
CREATE INDEX patient_sex ON patient(patient_sex(4));
CREATE INDEX patient_bitrh_date ON patient(patient_birth_date);

CREATE TABLE study(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    patient_fk BIGINT NOT NULL,
    study_id VARCHAR(16) NOT NULL,
    study_uid VARCHAR(64) NOT NULL,
    study_date_time TIMESTAMP,
    accession_no VARCHAR(16),
    study_desc VARCHAR(64),
    num_series INT NOT NULL,
    num_instance INT NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT patient_fk FOREIGN KEY (patient_fk) REFERENCES patient(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE UNIQUE INDEX study_uid ON study(study_uid(64));
CREATE INDEX study_id ON study(study_id(16));
CREATE INDEX study_date_time ON study(study_date_time);
CREATE INDEX accession_no ON study(accession_no(16));
CREATE INDEX created_time ON study(created_time);
CREATE INDEX updated_time ON study(updated_time);

CREATE TABLE series(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    study_fk BIGINT NOT NULL,
    series_uid VARCHAR(64) NOT NULL,
    series_no VARCHAR(16),
    modality VARCHAR(16),
    series_desc VARCHAR(64),
    institution VARCHAR(64),
    instance_num INT NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT study_fk FOREIGN KEY (study_fk) REFERENCES study(id) ON DELETE RESTRICT ON UPDATE RESTRICT    
);

CREATE UNIQUE INDEX series_uid ON series(series_uid(64));
CREATE INDEX series_no ON series(series_no(12));
CREATE INDEX modality ON series(modality(16));
CREATE INDEX institution ON series(institution(64));
CREATE INDEX created_time ON series(created_time);
CREATE INDEX updated_time ON series(updated_time);

CREATE TABLE instance(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    sop_class_uid VARCHAR(64) NOT NULL,
    sop_instance_uid VARCHAR(64) NOT NULL,
    retrieve_user_fk VARCHAR(32) NOT NULL,
    file_path VARCHAR(4096) NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT series_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT retrieve_user_fk FOREIGN KEY (retrieve_user_fk) REFERENCES user(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE UNIQUE INDEX sop_instance_uid ON instance(sop_instance_uid);

CREATE TABLE preprocess(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    pre_seg_mask_version VARCHAR(32),
    pre_seg_mask_file_path VARCHAR(4096),
    ai_cache_data_version VARCHAR(32),
    ai_cache_data_file VARCHAR(4096),
    CONSTRAINT series_prep_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE TABLE evaluation(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    lung_eva_version VARCHAR(32),
    lung_eva_file_path VARCHAR(4096),
    CONSTRAINT series_eva_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE TABLE annotation(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    annotation_type VARCHAR(32) NOT NULL,
    annotation_desc VARCHAR(256),
    file_path VARCHAR(4096) NOT NULL,
    annotation_user VARCHAR(32) NOT NULL,
    CONSTRAINT series_anno_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT annotation_user FOREIGN KEY (annotation_user) REFERENCES user(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);
"
echo creare DB success
