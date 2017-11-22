ver = debug
CC  = g++ -std=c++11 -Wall -Wno-unused-function

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/io
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

DATA_OBJS = mi_image_data.o mi_dicom_rt_struct.o
DATA_PARSER_OBJS = mi_meta_object_loader.o mi_dicom_exporter.o mi_dicom_loader.o mi_dicom_rt_struct_loader.o \
mi_nodule_set_parser.o mi_zlib_utils.o mi_jpeg_parser.o mi_targa_parser.o mi_mysql_db.o
PACS_OBJS = mi_dcm_scu.o mi_dcm_scp.o mi_worklist_info.o mi_pacs_communicator.o
OBJS = $(DATA_OBJS) $(DATA_PARSER_OBJS) $(PACS_OBJS)

TARGET = $(BIN_DIR)/libmedimg_io.so

EXPORT_HEADER = mi_io_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lboost_system -lboost_filesystem -lboost_thread -ljpeg \
-ldcmdata -ldcmimage -ldcmnet -loflog -lofstd -ldcmjpeg -lz -lmysqlcppconn -lmedimg_log -lmedimg_util -lmedimg_arithmetic

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

$(DATA_OBJS): %.o:%.cpp %.h 
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

$(DATA_PARSER_OBJS): %.o:%.cpp %.h $(DATA_OBJS)
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

#PACS OBJS
mi_dcm_scu.o: mi_dcm_scu.cpp mi_dcm_scu.h 
	$(CC) -c -fPIC $(CFLAGS) mi_dcm_scu.cpp $(ADDITINAL_INCLUDE_DIR)

mi_dcm_scp.o: mi_dcm_scp.cpp mi_dcm_scp.h 
	$(CC) -c -fPIC $(CFLAGS) mi_dcm_scp.cpp $(ADDITINAL_INCLUDE_DIR)

mi_worklist_info.o: mi_worklist_info.cpp mi_worklist_info.h 
	$(CC) -c -fPIC $(CFLAGS) mi_worklist_info.cpp $(ADDITINAL_INCLUDE_DIR)

mi_pacs_communicator.o: mi_pacs_communicator.cpp mi_pacs_communicator.h mi_worklist_info.o mi_dcm_scp.o mi_dcm_scu.o
	$(CC) -c -fPIC $(CFLAGS) mi_pacs_communicator.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
