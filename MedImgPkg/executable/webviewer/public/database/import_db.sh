cd ~/
if [ ! -d med_img_db ]; then
    mkdir med_img_db
fi
/home/wangrui22/program/git/dust-storm/MedImgPkg/bin/dcmdbimporter -u root -i 172.23.236.219:3306 -d med_img_db -r \
/home/wangrui22/mysql_db/med_img_cache_db/ -m /home/wangrui22/med_img_db/