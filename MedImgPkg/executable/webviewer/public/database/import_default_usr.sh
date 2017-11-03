mysql -u root -p$1 -e"
use med_img_db;
delete from usr where name='usr0';
delete from usr where name='usr1';
delete from usr where name='usr2';
delete from usr where name='usr3';
delete from usr where name='usr4';
delete from usr where name='usr5';
delete from usr where name='usr6';
delete from usr where name='usr7';
delete from usr where name='usr8';
delete from usr where name='usr9';
delete from usr where name='usr10';
insert into usr(name,role,password) value('usr0',0,'1');
insert into usr(name,role,password) value('usr1',0,'1');
insert into usr(name,role,password) value('usr2',0,'1');
insert into usr(name,role,password) value('usr3',0,'1');
insert into usr(name,role,password) value('usr4',0,'1');
insert into usr(name,role,password) value('usr5',0,'1');
insert into usr(name,role,password) value('usr6',0,'1');
insert into usr(name,role,password) value('usr7',0,'1');
insert into usr(name,role,password) value('usr8',0,'1');
insert into usr(name,role,password) value('usr9',0,'1');
insert into usr(name,role,password) value('usr10',0,'1');
select * from usr;
"