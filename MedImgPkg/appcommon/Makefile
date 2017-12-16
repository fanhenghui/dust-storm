ver = debug
CC  = g++ -std=c++11 -Wall -Wno-unused-function

ifeq ($(ver) , release)
CFLAGS += -O3 -fopenmp
else
CFLAGS += -g -fopenmp
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/appcommon
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

MODEL_OBJS = mi_model_crosshair.o mi_model_annotation.o mi_model_dbs_status.o mi_model_pacs.o mi_model_anonymization.o
OBSERVER_OBJS = mi_ob_annotation_segment.o mi_ob_annotation_statistic.o mi_ob_annotation_list.o
CMD_HANDLER_OBJS = mi_be_cmd_handler_fe_operation.o mi_be_cmd_handler_fe_heartbeat.o mi_be_cmd_handler_fe_ready.o \
mi_be_cmd_handler_fe_shutdown.o mi_be_cmd_handler_fe_anonymization.o mi_be_cmd_handler_fe_db_retrieve.o mi_be_cmd_handler_fe_play_vr.o \
mi_be_cmd_handler_db_send_error.o mi_be_cmd_handler_db_send_ai_evaluation.o mi_be_cmd_handler_db_send_end_signal.o \
mi_be_cmd_handler_db_send_preprocess_mask.o mi_be_cmd_handler_db_send_dicom.o mi_be_cmd_handler_fe_back_to_worklist.o \
mi_be_cmd_handler_fe_pacs_fetch.o mi_be_cmd_handler_fe_pacs_retrieve.o mi_be_cmd_handler_db_pacs_fetch_result.o \
mi_be_cmd_handler_db_pacs_retrieve_result.o
OPERATION_OBJS = mi_be_operation_fe_mpr_paging.o mi_be_operation_fe_rotate.o mi_be_operation_fe_pan.o mi_be_operation_fe_zoom.o \
mi_be_operation_fe_windowing.o mi_be_operation_fe_resize.o mi_be_operation_fe_mpr_mask_overlay.o mi_be_operation_fe_switch_preset_windowing.o \
mi_be_operation_fe_switch_preset_vrt.o mi_be_operation_fe_locate.o mi_be_operation_fe_downsample.o mi_be_operation_fe_annotation.o mi_be_operation_fe_fetch_ai_evaluation.o \
mi_be_operation_fe_adjust_evaluation_probability.o 
THREAD_MODEL_OBJS = mi_app_cell.o mi_app_none_image_item.o mi_app_none_image.o mi_app_controller.o mi_app_thread_model.o mi_app_common_util.o

OBJS = $(MODEL_OBJS) $(OBSERVER_OBJS) $(CMD_HANDLER_OBJS) $(OPERATION_OBJS) $(THREAD_MODEL_OBJS)

TARGET = $(BIN_DIR)/libmedimg_appcommon.so

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lGL -lGLEW -lprotobuf\
-lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_renderalgo

$(TARGET): $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

$(MODEL_OBJS):%.o:%.cpp %.h
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)	

mi_app_common_util.o: mi_app_common_util.h mi_app_common_util.cpp mi_app_common_define.h $(MODEL_OBJS) 
	$(CC) -c -fPIC $(CFLAGS) mi_app_common_util.cpp $(ADDITINAL_INCLUDE_DIR)

$(OBSERVER_OBJS):%.o:%.cpp %.h $(MODEL_OBJS) 
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)		

$(CMD_HANDLER_OBJS):%.o:%.cpp %.h $(MODEL_OBJS) mi_app_controller.o mi_app_common_util.o
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

$(OPERATION_OBJS):%.o:%.cpp %.h $(MODEL_OBJS) mi_app_controller.o mi_app_common_util.o
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

mi_app_cell.o: mi_app_cell.h mi_app_cell.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_app_cell.cpp $(ADDITINAL_INCLUDE_DIR)		

mi_app_none_image_item.o: mi_app_none_image_item.cpp mi_app_none_image_item.h $(MODEL_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_app_none_image_item.cpp $(ADDITINAL_INCLUDE_DIR)

mi_app_none_image.o: mi_app_none_image.cpp mi_app_none_image.h mi_app_none_image_item.o
	$(CC) -c -fPIC $(CFLAGS) mi_app_none_image.cpp $(ADDITINAL_INCLUDE_DIR)

mi_app_thread_model.o: mi_app_thread_model.cpp mi_app_thread_model.h mi_app_none_image.o mi_app_cell.o
	$(CC) -c -fPIC $(CFLAGS) mi_app_thread_model.cpp $(ADDITINAL_INCLUDE_DIR)

mi_app_controller.o: mi_app_controller.cpp mi_app_controller.h mi_app_thread_model.o mi_app_none_image.o mi_app_cell.o
	$(CC) -c -fPIC $(CFLAGS) mi_app_controller.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
