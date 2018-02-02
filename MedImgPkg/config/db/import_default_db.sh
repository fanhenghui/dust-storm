mysql -u root -p$1 -e"
use med_img_db;
INSERT INTO preprocess_type(prep_name, prep_desc) VALUES(\"init_segment_mask\",\"initialized segment mask(.rle)\");
INSERT INTO preprocess_type(prep_name, prep_desc) VALUES(\"lung_ai_intermediate_data\",\"lung ai algorithm intermediate data(.npy)\");

INSERT INTO evaluation_type(eva_name,eva_desc) VALUES(\"lung_nodule\",\"lung nodule evaluation(.csv)\");

INSERT INTO role(name) VALUES(\"chief physician\");
INSERT INTO role(name) VALUES(\"associate chief physician\");
INSERT INTO role(name) VALUES(\"attending physician\");
INSERT INTO role(name) VALUES(\"resident physician\");
"