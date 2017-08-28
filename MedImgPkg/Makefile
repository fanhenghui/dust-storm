MAKE = make
ver = debug

subsystem:
	-mkdir bin
	$(MAKE) -C ./util
	$(MAKE) -C ./glresource
	$(MAKE) -C ./arithmetic
	$(MAKE) -C ./io
	$(MAKE) -C ./executable/shaderencoder
	$(MAKE) -C ./renderalgo
	$(MAKE) -C ./appcommon
	$(MAKE) -C ./ut/util_ut
	$(MAKE) -C ./ut/glresource_ut
	$(MAKE) -C ./ut/renderalgo_ut
	$(MAKE) -C ./ut/io_ut
	$(MAKE) -C ./ut/protocol_ut
	$(MAKE) -C ./executable/reviewserver
	$(MAKE) -C ./executable/dcmdbimporter

PHONY.:clean
clean:
	$(MAKE) clean -C ./util
	$(MAKE) clean -C ./glresource
	$(MAKE) clean -C ./arithmetic
	$(MAKE) clean -C ./io
	$(MAKE) clean -C ./executable/shaderencoder
	$(MAKE) clean -C ./renderalgo
	$(MAKE) clean -C ./appcommon
	$(MAKE) clean -C ./ut/util_ut
	$(MAKE) clean -C ./ut/glresource_ut
	$(MAKE) clean -C ./ut/renderalgo_ut
	$(MAKE) clean -C ./ut/io_ut
	$(MAKE) clean -C ./ut/protocol_ut
	$(MAKE) clean -C ./executable/reviewserver
	$(MAKE) clean -C ./executable/dcmdbimporter
	-rm -r bin