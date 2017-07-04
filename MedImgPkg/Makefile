MAKE = make
ver = release

subsystem:
	-mkdir bin
	$(MAKE) -C ./MedImgUtil
	$(MAKE) -C ./MedImgGLResource
	$(MAKE) -C ./MedImgAppCommon
	$(MAKE) -C ./UT/MedImgUtil_UT
	$(MAKE) -C ./UT/MedImgGLResource_UT

	-export LD_LIBRARY_PATH=/home/	wr/program/git/dust-storm/MedImgPkg/bin 　　 

PHONY.:clean
clean:
	$(MAKE) clean -C ./MedImgUtil
	$(MAKE) clean -C ./UT/MedImgGLResource
	$(MAKE) clean -C ./MedImgAppCommon
	$(MAKE) clean -C ./UT/MedImgUtil_UT
	$(MAKE) clean -C ./UT/MedImgGLResource_UT