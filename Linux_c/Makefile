
export ver = debug

MAKE = make

define CREARE_DIR
	-mkdir bin
	-cd ./bin  -mkdir debug  -mkdir release
	-mkdir lib
	-cd ./lib  -mkdir debug  -mkdir release
endef

subsystem:
	$(CREARE_DIR)
	$(MAKE) -C ./classes
	$(MAKE) -C ./main

.PHONY:clean
clean:
	make clean -C ./classes
	make clean -C ./main