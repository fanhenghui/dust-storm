#recreate medical image database (med_img_db)
mysql -u root -p$1 -e"
drop database if exists med_img_db;
create database med_img_db;
show databases;
use med_img_db;

CREATE TABLE role(
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
    name VARCHAR(64) CHARACTER SET UTF8 NOT NULL
);

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
    patient_birth_date DATE,
    md5 VARCHAR(32) NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX md5 ON patient(md5);
CREATE INDEX patient_id ON patient(patient_id(64));
CREATE INDEX patient_name ON patient(patient_name(64));
CREATE INDEX patient_sex ON patient(patient_sex(4));
CREATE INDEX patient_birth_date ON patient(patient_birth_date);

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
    num_instance INT NOT NULL,
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
    retrieve_user_fk VARCHAR(32),
    file_path VARCHAR(4096) NOT NULL,
    file_size BIGINT NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT series_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT instance_retrieve_user_fk FOREIGN KEY (retrieve_user_fk) REFERENCES user(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE UNIQUE INDEX sop_instance_uid ON instance(sop_instance_uid(64));

CREATE TABLE preprocess_type (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    prep_name VARCHAR(32) NOT NULL,
    prep_desc VARCHAR(64),
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX prep_name ON preprocess_type(prep_name(32));

CREATE TABLE preprocess(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    prep_type_fk INT NOT NULL,
    retrieve_user_fk VARCHAR(32) NOT NULL,
    version VARCHAR(32),
    file_path VARCHAR(4096),
    file_size BIGINT,
    CONSTRAINT series_preprocess_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT prep_type_fk FOREIGN KEY (prep_type_fk) REFERENCES preprocess_type(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT preprocess_retrieve_user_fk FOREIGN KEY (retrieve_user_fk) REFERENCES user(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE TABLE evaluation_type (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    eva_name VARCHAR(32) NOT NULL,
    eva_desc VARCHAR(64) CHARACTER SET UTF8,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX eva_name ON evaluation_type(eva_name(32));

CREATE TABLE evaluation(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    eva_type_fk INT NOT NULL,
    retrieve_user_fk VARCHAR(32),
    version VARCHAR(32),
    file_path VARCHAR(4096),
    file_size BIGINT,
    CONSTRAINT series_eva_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT eva_type_fk FOREIGN KEY (eva_type_fk) REFERENCES evaluation_type(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT evaluation_retrieve_user_fk FOREIGN KEY (retrieve_user_fk) REFERENCES user(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE TABLE annotation(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    anno_type_fk INT NOT NULL,
    annotation_user VARCHAR(32),
    anno_desc VARCHAR(64) CHARACTER SET UTF8,
    file_path VARCHAR(4096) NOT NULL,
    file_size BIGINT NOT NULL,
    CONSTRAINT series_anno_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT annotation_user FOREIGN KEY (annotation_user) REFERENCES user(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT anno_type_fk FOREIGN KEY (anno_type_fk) REFERENCES evaluation_type(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);

CREATE INDEX anno_desc ON annotation(anno_desc(64));

show tables;
\\! echo \"\n+--------------------+\n| table: role \n+--------------------+\n\";
desc role;
\\! echo \"\n+--------------------+\n| table: user \n+--------------------+\n\";
desc user;
\\! echo \"\n+--------------------+\n| table: patient \n+--------------------+\n\";
desc patient;
\\! echo \"\n+--------------------+\n| table: study \n+--------------------+\n\";
desc study;
\\! echo \"\n+--------------------+\n| table: series \n+--------------------+\n\";
desc series;
\\! echo \"\n+--------------------+\n| table: instance \n+--------------------+\n\";
desc instance;
\\! echo \"\n+-----------------------+\n| table: preprocess_type \n+-----------------------+\n\";
desc preprocess_type;
\\! echo \"\n+--------------------+\n| table: preprocess \n+--------------------+\n\";
desc preprocess;
\\! echo \"\n+-----------------------+\n| table: evaluation_type \n+-----------------------+\n\";
desc evaluation_type;
\\! echo \"\n+--------------------+\n| table: evaluation \n+--------------------+\n\";
desc evaluation;
\\! echo \"\n+--------------------+\n| table: annotation \n+--------------------+\n\";
desc annotation;
"
echo creare DB success
