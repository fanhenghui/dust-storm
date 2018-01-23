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
delete from usr where name='wr';
insert into usr(name,role,password,online) value('usr0',0,'1',0);
insert into usr(name,role,password,online) value('usr1',0,'1',0);
insert into usr(name,role,password,online) value('usr2',0,'1',0);
insert into usr(name,role,password,online) value('usr3',0,'1',0);
insert into usr(name,role,password,online) value('usr4',0,'1',0);
insert into usr(name,role,password,online) value('usr5',0,'1',0);
insert into usr(name,role,password,online) value('usr6',0,'1',0);
insert into usr(name,role,password,online) value('usr7',0,'1',0);
insert into usr(name,role,password,online) value('usr8',0,'1',0);
insert into usr(name,role,password,online) value('usr9',0,'1',0);
insert into usr(name,role,password,online) value('usr10',0,'1',0);
insert into usr(name,role,password,online) value('wr',0,'1',0);
select * from usr;
"