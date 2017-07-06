MAKE = make
ver = debug

subsystem:
	-mkdir bin
	$(MAKE) -C ./MedImgUtil
	$(MAKE) -C ./MedImgGLResource
	$(MAKE) -C ./MedImgArithmetic
	$(MAKE) -C ./MedImgIO
	$(MAKE) -C ./MedImgAppCommon
	$(MAKE) -C ./Executable/MedImgShaderEncoder
	$(MAKE) -C ./MedImgRenderAlgorithm
	$(MAKE) -C ./UT/MedImgUtil_UT
	$(MAKE) -C ./UT/MedImgGLResource_UT
	$(MAKE) -C ./Executable/ReviewServer

	-export LD_LIBRARY_PATH=/home/	wr/program/git/dust-storm/MedImgPkg/bin 　　 

PHONY.:clean
clean:
	$(MAKE) clean -C ./MedImgUtil
	$(MAKE) clean -C ./MedImgGLResource
	$(MAKE) clean -C ./MedImgArithmetic
	$(MAKE) clean -C ./MedImgIO
	$(MAKE) clean -C ./MedImgAppCommon
	$(MAKE) clean -C ./Executable/MedImgShaderEncoder
	$(MAKE) clean -C ./MedImgRenderAlgorithm
	$(MAKE) clean -C ./UT/MedImgUtil_UT
	$(MAKE) clean -C ./UT/MedImgGLResource_UT
	$(MAKE) clean -C ./Executable/ReviewServer
	-rm -r bin