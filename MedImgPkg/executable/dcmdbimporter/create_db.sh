mysql -u root -p$1 -e"
drop database if exists med_img_cache_db;;
create database med_img_cache_db;
show databases;
use med_img_cache_db;
create table img_tbl(series_id VARCHAR(300) NOT NULL, study_id VARCHAR(300) NOT NULL, patient_name VARCHAR(100) , patient_id VARCHAR(100) , modality VARCHAR(50) NOT NULL, path VARCHAR(4096) , PRIMARY KEY (series_id) );
"
echo creare DB success