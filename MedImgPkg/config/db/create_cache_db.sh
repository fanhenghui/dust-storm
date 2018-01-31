mysql -u root -p$1 -e"
drop database if exists med_img_cache_db;
create database med_img_cache_db;
show databases;
use med_img_cache_db;

CREATE TABLE series(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_uid VARCHAR(64) NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX series_uid ON series(series_uid(64));
CREATE INDEX created_time ON series(created_time);
CREATE INDEX updated_time ON series(updated_time);

CREATE TABLE instance(
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    series_fk BIGINT NOT NULL,
    sop_instance_uid VARCHAR(64) NOT NULL,
    file_path VARCHAR(4096) NOT NULL,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT series_fk FOREIGN KEY (series_fk) REFERENCES series(id) ON DELETE RESTRICT ON UPDATE RESTRICT
);
"
echo creare Cache DB success