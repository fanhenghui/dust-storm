mysql -uroot -p$1 -e"
use med_img_db;
DELETE FROM annotation;
DELETE FROM evaluation;
DELETE FROM evaluation_type;
DELETE FROM preprocess_type;
DELETE FROM preprocess;
DELETE FROM instance;
DELETE FROM series;
DELETE FROM study;
DELETE FROM patient;
DELETE FROM user;
DELETE FROM role;
\! echo clear med_img_db success.;
"