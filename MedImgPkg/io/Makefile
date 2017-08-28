ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/io
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_dicom_exporter.o mi_dicom_loader.o mi_dicom_rt_struct_loader.o mi_dicom_rt_struct.o \
mi_image_data.o mi_meta_object_loader.o mi_nodule_set_parser.o mi_nodule_set.o mi_zlib_utils.o \
mi_dcm_scu.o mi_dcm_scp.o mi_worklist_info.o mi_pacs_communicator.o
TARGET = $(BIN_DIR)/libmedimg_io.so

EXPORT_HEADER = mi_io_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lmedimg_arithmetic -lboost_system -lboost_filesystem -lboost_thread -ldcmdata -ldcmimage -ldcmjpeg -ldcmnet -lz

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_dicom_loader.o: $(EXPORT_HEADER) mi_dicom_loader.h mi_dicom_loader.cpp mi_image_data_header.h mi_image_data.o
	$(CC) -c -fPIC $(CFLAGS) mi_dicom_loader.cpp $(ADDITINAL_INCLUDE_DIR)

mi_dicom_exporter.o: $(EXPORT_HEADER) mi_dicom_exporter.h mi_dicom_exporter.cpp mi_image_data.o 
	$(CC) -c -fPIC $(CFLAGS) mi_dicom_exporter.cpp $(ADDITINAL_INCLUDE_DIR)

mi_image_data.o: $(EXPORT_HEADER) mi_image_data.h mi_image_data.cpp mi_io_define.h
	$(CC) -c -fPIC $(CFLAGS) mi_image_data.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_zlib_utils.o: $(EXPORT_HEADER) mi_zlib_utils.h mi_zlib_utils.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_zlib_utils.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_meta_object_loader.o: $(EXPORT_HEADER) mi_meta_object_loader.h mi_meta_object_loader.cpp mi_image_data_header.h mi_image_data.o mi_zlib_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_meta_object_loader.cpp $(ADDITINAL_INCLUDE_DIR)

mi_dicom_rt_struct.o: $(EXPORT_HEADER) mi_dicom_rt_struct.h mi_dicom_rt_struct.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_dicom_rt_struct.cpp $(ADDITINAL_INCLUDE_DIR)

mi_dicom_rt_struct_loader.o: $(EXPORT_HEADER) mi_dicom_rt_struct_loader.h mi_dicom_rt_struct_loader.cpp mi_dicom_rt_struct.o
	$(CC) -c -fPIC $(CFLAGS) mi_dicom_rt_struct_loader.cpp $(ADDITINAL_INCLUDE_DIR)

mi_nodule_set_parser.o: $(EXPORT_HEADER) mi_nodule_set_parser.h mi_nodule_set_parser.cpp mi_nodule_set.o
	$(CC) -c -fPIC $(CFLAGS) mi_nodule_set_parser.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_nodule_set.o: $(EXPORT_HEADER) mi_nodule_set.h mi_nodule_set.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_nodule_set.cpp $(ADDITINAL_INCLUDE_DIR)

mi_pacs_communicator.o: $(EXPORT_HEADER) mi_pacs_communicator.h mi_pacs_communicator.cpp mi_dcm_scu.o mi_dcm_scp.o mi_worklist_info.o
	$(CC) -c -fPIC $(CFLAGS) mi_pacs_communicator.cpp $(ADDITINAL_INCLUDE_DIR)

mi_worklist_info.o: $(EXPORT_HEADER) mi_worklist_info.h mi_worklist_info.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_worklist_info.cpp $(ADDITINAL_INCLUDE_DIR)

mi_dcm_scu.o: $(EXPORT_HEADER) mi_dcm_scu.h mi_dcm_scu.cpp mi_worklist_info.o
	$(CC) -c -fPIC $(CFLAGS) mi_dcm_scu.cpp $(ADDITINAL_INCLUDE_DIR)

mi_dcm_scp.o: $(EXPORT_HEADER) mi_dcm_scp.h mi_dcm_scp.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_dcm_scp.cpp $(ADDITINAL_INCLUDE_DIR)	

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
