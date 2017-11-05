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
	$(MAKE) -C ./executable/dcmdbimporter
	$(MAKE) -C ./executable/extractmask
	$(MAKE) -C ./executable/compressmask
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
	$(MAKE) clean -C ./executable/dcmdbimporter
	$(MAKE) clean -C ./executable/extractmask
	$(MAKE) clean -C ./executable/compressmask
	$(MAKE) clean -C ./ut/util_ut
	$(MAKE) clean -C ./ut/algorithm_ut
	$(MAKE) clean -C ./ut/glresource_ut
	$(MAKE) clean -C ./ut/renderalgo_ut
	$(MAKE) clean -C ./ut/io_ut
	$(MAKE) clean -C ./ut/protocol_ut
	-rm -r bin

ip = 127.0.0.1
des
PHONY.:install
install: 
	echo install path : $(DESTDIR)
	echo ip: $(ip)
	#create main direction
	test -d '$(DESTDIR)/med_img' || mkdir '$(DESTDIR)/med_img'
	#create bin direction
	test -d '$(DESTDIR)/med_img/bin' || mkdir '$(DESTDIR)/med_img/bin'
	#create webviewer direction
	test -d '$(DESTDIR)/med_img/webviewer' || mkdir ''
	cd -
	cp ./bin/*.so $(DESTDIR)/med_img/bin
	cp ./bin/*ut $(DESTDIR)/med_img/bin
	cp ./bin/review_server $(DESTDIR)/med_img/bin
	cp -r ./config $(DESTDIR)/med_img/