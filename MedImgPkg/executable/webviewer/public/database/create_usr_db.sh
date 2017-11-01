mysql -u root -p$1 -e"
drop database if exists med_img_usr_db;
create database med_img_usr_db;
show databases;
use med_img_usr_db;
create table usr_tbl(name VARCHAR(300) NOT NULL, role INT NOT NULL, password VARCHAR(30) NOT NULL, PRIMARY KEY (name));
"
echo creare Usr DB success