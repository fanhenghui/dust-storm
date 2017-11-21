ver = debug
CC  = g++ -std=c++11 -Wall -Wno-unused-function

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g 
endif

DEFS = -DBOOST_LOG_DYN_LINK -D_DEBUG
CFLAGS += $(DEFS)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/arithmetic
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_vector2.o mi_vector3.o mi_point2.o mi_point3.o mi_matrix4.o \
mi_vector2f.o mi_vector3f.o mi_vector4f.o mi_matrix4f.o \
mi_camera_base.o mi_ortho_camera.o mi_perspective_camera.o mi_track_ball.o \
mi_arithmetic_utils.o \
mi_aabb.o mi_ellipsoid.o mi_line_segment.o mi_line.o mi_plane.o mi_triangle.o mi_intersection_test.o \
mi_rsa_utils.o mi_volume_statistician.o mi_run_length_operator.o mi_connected_domain_analysis.o

TARGET = $(BIN_DIR)/libmedimg_arithmetic.so

EXPORT_HEADER = mi_arithmetic_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log_setup -lboost_log -lmedimg_log -lmedimg_util  -lmbedcrypto -lmbedtls -lmbedx509 

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

$(OBJS): %.o:%.cpp %.h 
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR) 

.PHONY : clean
clean:
	-rm *.o $(TARGET)
