ver = debug

CC = g++ -std=c++11 -Wall

MODULE_NAME = MedImgCommon
ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/$(MODULE_NAME)
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

ifeq ($(ver) , debug)
CFLAGS += -c -g -fPIC
else
CFLAGS += -c -fPIC
endif

TARGET = $(BIN_DIR)/lib$(MODULE_NAME).so
OBJS = mi_common_exception.o mi_concurrency.o mi_configuration.o mi_model_interface.o


EXPORT_HEADER = mi_common_stdafx.h 