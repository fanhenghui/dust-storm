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
	$(MAKE) -C ./MedImgAppCommon
	$(MAKE) -C ./UT/MedImgUtil_UT
	$(MAKE) -C ./UT/MedImgGLResource_UT
	$(MAKE) -C ./UT/MedImgRenderAlgorithm_UT
	$(MAKE) -C ./UT/MedImgIO_UT
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
	$(MAKE) clean -C ./MedImgAppCommon
	$(MAKE) clean -C ./UT/MedImgUtil_UT
	$(MAKE) clean -C ./UT/MedImgGLResource_UT
	$(MAKE) clean -C ./UT/MedImgRenderAlgorithm_UT
	$(MAKE) clean -C ./UT/MedImgIO_UT
	$(MAKE) clean -C ./executable/reviewserver
	$(MAKE) clean -C ./executable/dcmdbimporter
	-rm -r bin