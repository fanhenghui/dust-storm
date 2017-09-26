ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

PROTOCO_LIB_DIR = /usr/local/lib
GOOGLE_INCLUDE_DIR = /usr/local/include/google/
PROTOCO_INCLUDE_DIR = /usr/local/include/google/protobuf

VPATH = $(INCLUDE_ROOT_DIR)

OBJS = mi_review_config.o mi_review_controller.o mi_none_image_item.o mi_review_none_image.o \
mi_operation_mpr_paging.o mi_operation_rotate.o mi_operation_pan.o mi_operation_zoom.o \
mi_operation_windowing.o mi_operation_resize.o mi_operation_init.o mi_operation_annotation.o \
mi_search_worklist_command_handler.o mi_mpr_play_command_handler.o mi_vr_play_command_handler.o

TARGET = $(BIN_DIR)/review_server

COMMON_HEADER = mi_review_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(GOOGLE_INCLUDE_DIR) -I$(PROTOCO_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(PROTOCO_LIB_DIR) -lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_appcommon -lmedimg_renderalgo \
-lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lGL -lGLEW -lprotobuf

all: pre-build main-build

pre-build:
	sh protoc-gen.sh

main-build: $(TARGET)

EXPORT_HEADER = mi_review_common.h

$(TARGET): mi_message.pb.o mi_review_main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) mi_message.pb.o mi_review_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_review_main.o: $(EXPORT_HEADER)  mi_review_main.cpp  mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_review_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_message.pb.o: mi_message.pb.h mi_message.pb.cc 
	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

$(OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

# $(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so $(BIN_DIR)/libmedimg_appcommon.so $(BIN_DIR)/libmedimg_renderalgo.so
# 	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES) $(ADDITINAL_INCLUDE_DIR)

# mi_review_main.o: $(EXPORT_HEADER)  mi_review_main.cpp  mi_review_controller.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_review_main.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_search_worklist_command_handler.o: $(EXPORT_HEADER) mi_search_worklist_command_handler.h mi_search_worklist_command_handler.cpp \
# mi_review_controller.o mi_review_config.o mi_message.pb.o
# 	$(CC) -c -fPIC ${CFLAGS} mi_search_worklist_command_handler.cpp ${ADDITINAL_INCLUDE_DIR}

# mi_mpr_play_command_handler.o: $(EXPORT_HEADER) mi_mpr_play_command_handler.h mi_mpr_play_command_handler.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_mpr_play_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_vr_play_command_handler.o: $(EXPORT_HEADER) mi_vr_play_command_handler.h mi_vr_play_command_handler.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_vr_play_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_init.o: $(EXPORT_HEADER) mi_operation_init.h mi_operation_init.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_init.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_mpr_paging.o: $(EXPORT_HEADER) mi_operation_mpr_paging.h mi_operation_mpr_paging.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_mpr_paging.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_rotate.o: $(EXPORT_HEADER) mi_operation_rotate.h mi_operation_rotate.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_rotate.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_zoom.o: $(EXPORT_HEADER) mi_operation_zoom.h mi_operation_zoom.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_zoom.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_pan.o: $(EXPORT_HEADER) mi_operation_pan.h mi_operation_pan.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_pan.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_windowing.o: $(EXPORT_HEADER) mi_operation_windowing.h mi_operation_windowing.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_windowing.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_resize.o: $(EXPORT_HEADER) mi_operation_resize.h mi_operation_resize.cpp \
# mi_review_controller.o mi_message.pb.o
# 	$(CC) -c -fPIC $(CFLAGS) mi_operation_resize.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_operation_annotation.o: $(EXPORT_HEADER) mi_operation_annotation.h mi_operation_annotation.cpp \
# mi_review_controller.o mi_review_config.o mi_message.pb.o
# 	$(CC) -c -fPIC ${CFLAGS} mi_operation_annotation.cpp ${ADDITINAL_INCLUDE_DIR}

# mi_review_controller.o: $(EXPORT_HEADER) mi_review_controller.h mi_review_controller.cpp
# 	$(CC) -c -fPIC $(CFLAGS) mi_review_controller.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_review_config.o: $(EXPORT_HEADER) mi_review_config.h mi_review_config.cpp
# 	$(CC) -c -fPIC $(CFLAGS) mi_review_config.cpp $(ADDITINAL_INCLUDE_DIR)

# mi_message.pb.o: mi_message.pb.h mi_message.pb.cc 
# 	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
	-rm *.pb.cc *.pb.h

