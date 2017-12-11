ver = debug
CC  = g++ -std=c++11 -Wall -fopenmp

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

ROOT = ../../
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_extract_mask.o mi_extract_mask_vis.o mi_main.o
TARGET = $(BIN_DIR)/extractmask

COMMON_HEADER = mi_extract_mask_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) 
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lboost_system -lboost_filesystem -lboost_thread -lpugixml -lGL -lGLEW -lGLU -lglut -lmedimg_util -lmedimg_arithmetic -lmedimg_io 

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_extract_mask.o: mi_extract_mask.cpp 
	$(CC) -c $(CFLAGS) mi_extract_mask.cpp $(ADDITINAL_INCLUDE_DIR)

mi_extract_mask_vis.o: mi_extract_mask_vis.cpp mi_extract_mask.o
	$(CC) -c $(CFLAGS) mi_extract_mask_vis.cpp $(ADDITINAL_INCLUDE_DIR)

mi_main.o: mi_main.cpp mi_extract_mask.o mi_extract_mask_vis.o
	$(CC) -c $(CFLAGS) mi_main.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)