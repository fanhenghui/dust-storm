#recreate medical image database (med_img_db)
#table0 : usr;
#table1 : dcm_series;
#table2 : annotations;
#annotation name : series.usr.usr_name
mysql -u root -p$1 -e"
drop database if exists med_img_db;
create database med_img_db;
show databases;
use med_img_db;

create table usr(ID INT NOT NULL AUTO_INCREMENT, name VARCHAR(30) NOT NULL, role TINYINT NOT NULL, online BIT NOT NULL, password VARCHAR(30) NOT NULL, PRIMARY KEY (ID));

create table dcm_series(series_id VARCHAR(300) NOT NULL, 
study_id VARCHAR(300) NOT NULL, 
patient_name VARCHAR(100) , 
patient_id VARCHAR(100) , 
modality VARCHAR(50) NOT NULL, 
size_mb FLOAT NOT NULL,
dcm_path VARCHAR(4096) NOT NULL,
preprocess_mask_path VARCHAR(4096),
annotation_ai_path VARCHAR(4096), 
ai_intermediate_data_path VARCHAR(4096), 
PRIMARY KEY (series_id));

create table annotations(name VARCHAR(300) NOT NULL,
series_id VARCHAR(300) NOT NULL,
usr_name VARCHAR(80) NOT NULL,
annotation_usr_path VARCHAR(4096) NOT NULL);
"
echo creare Usr DB success
