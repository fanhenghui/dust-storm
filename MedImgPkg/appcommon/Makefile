ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/appcommon
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_app_database.o mi_app_cell.o mi_operation_factory.o mi_app_none_image_item.o mi_app_none_image.o \
mi_app_controller.o mi_app_thread_model.o \
mi_cmd_handler_operating.o mi_cmd_handler_heartbeat.o mi_cmd_handler_fe_ready.o mi_cmd_handler_shutdown.o \
mi_cmd_handler_mpr_play.o mi_cmd_handler_vr_play.o \
mi_operation_mpr_paging.o mi_operation_rotate.o mi_operation_pan.o mi_operation_zoom.o \
mi_operation_windowing.o mi_operation_resize.o \
mi_model_annotation.o mi_ob_annotation_segment.o mi_ob_annotation_statistic.o mi_operation_annotation.o mi_ob_annotation_none_image.o

PROTOC = mi_message.pb.o

TARGET = $(BIN_DIR)/libmedimg_appcommon.so

EXPORT_HEADER = mi_app_common_export.h mi_app_common_define.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_renderalgo \
-lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lmysqlcppconn -lGL -lGLEW -lprotobuf

all: pre-build main-build

pre-build:
	sh protoc-gen.sh

main-build: $(TARGET)

$(TARGET): $(PROTOC) $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(PROTOC) $(ADDITIANL_DEPENDENCIES)

$(PROTOC): mi_message.pb.h mi_message.pb.cc 
	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

$(OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)	

.PHONY : clean
clean:
	-rm *.o $(TARGET)
	-rm *.pb.cc *.pb.h
