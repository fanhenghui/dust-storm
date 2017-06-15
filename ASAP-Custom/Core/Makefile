#Core makefile

#version (default debug)
ver = debug

#compiler
CC = g++ -std=c++11

#file direction
MODULE_NAME = core
INCLUDE_DIR = ../include
LIB_DIR = ../lib

VPATH = $(INCLUDE_DIR):$(INCLUDE_DIR)/$(MODULE_NAME)
#vpath %.h $(INCLUDE_DIR):$(INCLUDE_DIR)/$(MODULE_NAME)

#debug/release cc
ifeq ($(ver) , debug)
CFLAGS += -c -g -fPIC
else
CFLAGS += -c -fPIC
endif

#macro
DEFS = CORE_STATIC_DEFINE 
CFLAGS += -D $(DEFS)

#target and obj
TARGET = $(LIB_DIR)/libcore.so
OBJS = filetools.o CmdLineProgressMonitor.o ImageSource.o stringconversion.o ProgressMonitor.o Box.o Point.o PathologyEnums.o 

EXPORT_HEADER = core_export.h

$(TARGET): $(OBJS) $(EXPORT_HEADER)
	$(CC) -shared -fpic -lboost_system -lboost_filesystem -lboost_regex -o $(TARGET) $(OBJS)

filetools.o : $(EXPORT_HEADER) filetools.cpp filetools.h stringconversion.o
	$(CC) $(CFLAGS) filetools.cpp -I$(INCLUDE_DIR)  -I$(INCLUDE_DIR)/$(MODULE_NAME)

CmdLineProgressMonitor.o : $(EXPORT_HEADER) CmdLineProgressMonitor.cpp CmdLineProgressMonitor.h ProgressMonitor.o
	$(CC) $(CFLAGS) CmdLineProgressMonitor.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)
	
ImageSource.o : $(EXPORT_HEADER) ImageSource.cpp ImageSource.h PathologyEnums.o
	$(CC) $(CFLAGS) ImageSource.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)	

stringconversion.o : $(EXPORT_HEADER) stringconversion.cpp stringconversion.h 
	$(CC) $(CFLAGS) stringconversion.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

ProgressMonitor.o : $(EXPORT_HEADER) ProgressMonitor.cpp ProgressMonitor.h
	$(CC) $(CFLAGS) ProgressMonitor.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

Box.o : $(EXPORT_HEADER) Box.cpp Box.h
	$(CC) $(CFLAGS) Box.cpp -I$(INCLUDE_DIR)  -I$(INCLUDE_DIR)/$(MODULE_NAME)

Point.o : $(EXPORT_HEADER) Point.cpp Point.h
	$(CC) $(CFLAGS) Point.cpp -I$(INCLUDE_DIR)  -I$(INCLUDE_DIR)/$(MODULE_NAME)

PathologyEnums.o : $(EXPORT_HEADER) PathologyEnums.cpp PathologyEnums.h
	$(CC) $(CFLAGS) PathologyEnums.cpp -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/$(MODULE_NAME)


.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)





	



