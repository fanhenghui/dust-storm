MAKE = make
ver = debug

subsystem:
	-mkdir bin
	-export LD_LIBRARY_PATH=/home/wr/program/git/dust-storm/MedImgPkg/bin:/home/wr/program/git/dust-storm/MedImgPkg/:/usr/local/lib
	$(MAKE) -C ./MedImgUtil
	$(MAKE) -C ./MedImgGLResource
	$(MAKE) -C ./MedImgArithmetic
	$(MAKE) -C ./MedImgIO
	$(MAKE) -C ./Executable/MedImgShaderEncoder
	$(MAKE) -C ./MedImgRenderAlgorithm
	$(MAKE) -C ./MedImgAppCommon
	$(MAKE) -C ./UT/MedImgUtil_UT
	$(MAKE) -C ./UT/MedImgGLResource_UT
	$(MAKE) -C ./UT/MedImgRenderAlgorithm_UT
	$(MAKE) -C ./UT/MedImgIO_UT
	$(MAKE) -C ./Executable/ReviewServer

PHONY.:clean
clean:
	$(MAKE) clean -C ./MedImgUtil
	$(MAKE) clean -C ./MedImgGLResource
	$(MAKE) clean -C ./MedImgArithmetic
	$(MAKE) clean -C ./MedImgIO
	$(MAKE) clean -C ./Executable/MedImgShaderEncoder
	$(MAKE) clean -C ./MedImgRenderAlgorithm
	$(MAKE) clean -C ./MedImgAppCommon
	$(MAKE) clean -C ./UT/MedImgUtil_UT
	$(MAKE) clean -C ./UT/MedImgGLResource_UT
	$(MAKE) clean -C ./UT/MedImgRenderAlgorithm_UT
	$(MAKE) clean -C ./UT/MedImgIO_UT
	$(MAKE) clean -C ./Executable/ReviewServer
	-rm -r bin