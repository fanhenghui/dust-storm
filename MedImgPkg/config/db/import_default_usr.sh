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
delete from usr where name='wr';
insert into usr(ID,name,role,password) value('d8faf4f1cd5545e0a9338c511cfb6183','usr0',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('9a32fd77566a4e6788cd1da7bf496db7','usr1',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('6f15b3b155214d6a82a90340c2648291','usr2',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('d8f3cb7372b447adb16fe85b6ed747a8','usr3',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('8334b9efdf034eaabda1dbd4e9397702','usr4',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('ee599dcbaec347bf838a0ec00fd6fb66','usr5',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('97f02ecfec5949148286ca99b88f517c','usr6',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('9dc80f1603cb45e493d0fc4946a6a718','usr7',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('a5528bacf57640fa94dd527c3e4500ef','usr8',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('c2eb78d24e2d448db73ff5b44c8ec4c7','usr9',0,'c4ca4238a0b923820dcc509a6f75849b');
insert into usr(ID,name,role,password) value('50b5ea8ccb6046f1a632b34763149270','wr'  ,0,'c4ca4238a0b923820dcc509a6f75849b');
select * from usr;
"