cd ~/
if [ ! -d mysql_db ]; then
    mkdir mysql_db
fi
cd mysql_db
if [ ! -d med_img_cache_db ]; then
    mkdir med_img_cache_db
fi
./../../bin/dcmdbimporter -u root -i 127.0.0.1:3306 -d med_img_cache_db -r /home/wangrui22/data/ -m /home/wangrui22/mysql_db/med_img_cache_db/