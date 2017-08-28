ver = release
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)

else
CFLAGS += -g
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/shaderencoder

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lboost_system -lboost_filesystem

$(TARGET):  mi_shader_encoder.cpp
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET)  mi_shader_encoder.cpp  $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)
