ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ../../
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_extract_mask.o mi_extract_mask_vis.o mi_main.o
TARGET = $(BIN_DIR)/extract_mask

COMMON_HEADER = mi_extract_mask_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) 
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lboost_system -lboost_filesystem -lboost_thread -lpugixml -lGL -lGLEW -lGLU -lglut -lmedimg_util -lmedimg_arithmetic -lmedimg_io 

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)
$(OBJS):%.o:%.cpp $(COMMON_HEADER)
	$(CC) -c $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)