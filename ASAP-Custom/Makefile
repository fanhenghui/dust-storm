ver = debug

MAKE = make
subsystem:
	-mkdir lib
	$(MAKE) -C ./Core
	$(MAKE) -C ./IO/MultiResolutionImageInterface
	$(MAKE) -C ./Annotation
	$(MAKE) -C ./Executables/AnnoServer
#add debug/release

.PHONY : clean
clean:
	make clean -C ./Core
	make clean -C ./IO/MultiResolutionImageInterface
	make clean -C ./Annotation
	make clean -C ./Executables/AnnoServer

