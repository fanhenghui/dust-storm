MAKE = make
ver = debug

subsystem:
	-mkdir bin
	$(MAKE) -C ./log
	$(MAKE) -C ./util
	$(MAKE) -C ./glresource
	$(MAKE) -C ./arithmetic
	$(MAKE) -C ./io
	$(MAKE) -C ./executable/shaderencoder
	$(MAKE) -C ./renderalgo
	$(MAKE) -C ./appcommon
	$(MAKE) -C ./executable/reviewserver
	$(MAKE) -C ./executable/dbserver
	$(MAKE) -C ./executable/dcmdbimporter
	$(MAKE) -C ./executable/extractmask
	$(MAKE) -C ./executable/compressmask
	$(MAKE) -C ./ut/dbserver_ut
	$(MAKE) -C ./ut/appcommon_ut
	$(MAKE) -C ./ut/util_ut
	$(MAKE) -C ./ut/algorithm_ut
	$(MAKE) -C ./ut/glresource_ut
	$(MAKE) -C ./ut/renderalgo_ut
	$(MAKE) -C ./ut/io_ut
	$(MAKE) -C ./ut/protocol_ut

PHONY.:clean
clean:
	$(MAKE) clean -C ./log
	$(MAKE) clean -C ./util
	$(MAKE) clean -C ./glresource
	$(MAKE) clean -C ./arithmetic
	$(MAKE) clean -C ./io
	$(MAKE) clean -C ./executable/shaderencoder
	$(MAKE) clean -C ./renderalgo
	$(MAKE) clean -C ./appcommon
	$(MAKE) clean -C ./executable/reviewserver
	$(MAKE) clean -C ./executable/dbserver
	$(MAKE) clean -C ./executable/dcmdbimporter
	$(MAKE) clean -C ./executable/extractmask
	$(MAKE) clean -C ./executable/compressmask
	$(MAKE) clean -C ./ut/dbserver_ut
	$(MAKE) clean -C ./ut/appcommon_ut
	$(MAKE) clean -C ./ut/util_ut
	$(MAKE) clean -C ./ut/algorithm_ut
	$(MAKE) clean -C ./ut/glresource_ut
	$(MAKE) clean -C ./ut/renderalgo_ut
	$(MAKE) clean -C ./ut/io_ut
	$(MAKE) clean -C ./ut/protocol_ut
	-rm -r bin

DESTDIR = /usr/local
PHONY.:install
install: 
	echo install path : $(DESTDIR)
	echo ip: $(ip)
	#create main direction
	test -d '$(DESTDIR)/med_img' || mkdir '$(DESTDIR)/med_img'
	#create bin direction
	test -d '$(DESTDIR)/med_img/bin' || mkdir '$(DESTDIR)/med_img/bin'
	#create webviewer direction
	test -d '$(DESTDIR)/med_img/webviewer/public/medical_web' || mkdir -p '$(DESTDIR)/med_img/webviewer/public/medical_web'

	#cp app server
	cp -r ./bin/* $(DESTDIR)/med_img/bin
	cp -r ./config $(DESTDIR)/med_img/
	#cp web module
	cp -r ./executable/webviewer/node_modules  $(DESTDIR)/med_img/webviewer/node_modules
	cp ./executable/webviewer/install_npm.sh  $(DESTDIR)/med_img/webviewer/install_npm.sh
	cp ./executable/webviewer/nginx.conf  $(DESTDIR)/med_img/webviewer/nginx.conf
	cp ./executable/webviewer/package-lock.json  $(DESTDIR)/med_img/webviewer/package-lock.json
	#cp web server
	cp ./executable/webviewer/server.js  $(DESTDIR)/med_img/webviewer/server.js
	cp -r ./executable/webviewer/public/routes  $(DESTDIR)/med_img/webviewer/public/
	cp -r ./executable/webviewer/public/database  $(DESTDIR)/med_img/webviewer/public/
	cp -r ./executable/webviewer/public/be  $(DESTDIR)/med_img/webviewer/public/
	#cp web FE
	cp -r ./executable/webviewer/public/css  $(DESTDIR)/med_img/webviewer/public/medical_web
	cp -r ./executable/webviewer/public/data  $(DESTDIR)/med_img/webviewer/public/medical_web
	cp -r ./executable/webviewer/public/fonts  $(DESTDIR)/med_img/webviewer/public/medical_web
	cp -r ./executable/webviewer/public/images  $(DESTDIR)/med_img/webviewer/public/medical_web
	cp -r ./executable/webviewer/public/js  $(DESTDIR)/med_img/webviewer/public/medical_web
	cp -r ./executable/webviewer/public/views  $(DESTDIR)/med_img/webviewer/public/medical_web
	
	#install app server path
	echo $(DESTDIR)/med_img/bin/review_server > $(DESTDIR)/med_img/webviewer/public/be/be_path
	#install application config path
	test -d '$(DESTDIR)/med_img/webviewer/public/config' || mkdir -p '$(DESTDIR)/med_img/webviewer/public/config'
	echo $(DESTDIR)/med_img/config/app_config > $(DESTDIR)/med_img/webviewer/public/config/config_path
	echo install done