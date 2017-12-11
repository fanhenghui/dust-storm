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

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

DOUBLE_ARITHMETIC_OBJS = mi_vector2.o mi_vector3.o mi_point2.o mi_point3.o mi_matrix4.o
SEE_ARITHMETIC_OBJS = mi_vector2f.o mi_vector3f.o mi_vector4f.o mi_matrix4f.o
CAMERA_OBJS = mi_camera_base.o mi_ortho_camera.o mi_perspective_camera.o mi_track_ball.o
SHAPE_OBJS = mi_aabb.o mi_triangle.o
OBJS = $(SEE_ARITHMETIC_OBJS) $(DOUBLE_ARITHMETIC_OBJS) $(SHAPE_OBJS) $(CAMERA_OBJS) mi_arithmetic_utils.o \
mi_rsa_utils.o mi_volume_statistician.o mi_run_length_operator.o mi_connected_domain_analysis.o mi_intersection_test.o

TARGET = $(BIN_DIR)/libmedimg_arithmetic.so

EXPORT_HEADER = mi_arithmetic_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log_setup -lboost_log -lmedimg_log -lmedimg_util -lmbedcrypto -lmbedtls -lmbedx509 

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_log.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

#SEE arithmetic
mi_matrix4f.o: $(EXPORT_HEADER) mi_matrix4f.h mi_matrix4f.cpp mi_simd.h mi_vector3f.o mi_vector4f.o
	$(CC) -c -fPIC $(CFLAGS) mi_matrix4f.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector4f.o: $(EXPORT_HEADER) mi_vector4f.h mi_vector4f.cpp mi_simd.h mi_vector3f.o
	$(CC) -c -fPIC $(CFLAGS) mi_vector4f.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector3f.o: $(EXPORT_HEADER) mi_vector3f.h mi_vector3f.cpp mi_simd.h
	$(CC) -c -fPIC $(CFLAGS) mi_vector3f.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector2f.o: $(EXPORT_HEADER) mi_vector2f.h mi_vector2f.cpp mi_simd.h
	$(CC) -c -fPIC $(CFLAGS) mi_vector2f.cpp $(ADDITINAL_INCLUDE_DIR)

#Double arithmetic 
mi_matrix4.o: $(EXPORT_HEADER) mi_matrix4.h mi_matrix4.cpp mi_vector3.o mi_point3.o
	$(CC) -c -fPIC $(CFLAGS) mi_matrix4.cpp $(ADDITINAL_INCLUDE_DIR)

mi_point2.o: $(EXPORT_HEADER) mi_point2.h mi_point2.cpp mi_vector2.o
	$(CC) -c -fPIC $(CFLAGS) mi_point2.cpp $(ADDITINAL_INCLUDE_DIR)

mi_point3.o: $(EXPORT_HEADER) mi_point3.h mi_point3.cpp mi_vector3.o
	$(CC) -c -fPIC $(CFLAGS) mi_point3.cpp $(ADDITINAL_INCLUDE_DIR)
	
mi_vector3.o: $(EXPORT_HEADER) mi_vector3.h mi_vector3.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_vector3.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector2.o: $(EXPORT_HEADER) mi_vector2.h mi_vector2.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_vector2.cpp $(ADDITINAL_INCLUDE_DIR)

#Cameras
mi_track_ball.o: $(EXPORT_HEADER) mi_track_ball.h mi_track_ball.cpp $(DOUBLE_ARITHMETIC_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_track_ball.cpp $(ADDITINAL_INCLUDE_DIR)

mi_perspective_camera.o: $(EXPORT_HEADER) mi_perspective_camera.h mi_perspective_camera.cpp mi_camera_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_perspective_camera.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ortho_camera.o: $(EXPORT_HEADER) mi_ortho_camera.h mi_ortho_camera.cpp mi_camera_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_ortho_camera.cpp $(ADDITINAL_INCLUDE_DIR)

mi_camera_base.o: $(EXPORT_HEADER) mi_camera_base.h mi_camera_base.cpp $(DOUBLE_ARITHMETIC_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_camera_base.cpp $(ADDITINAL_INCLUDE_DIR)

#Shapes
$(SHAPE_OBJS): %.o:%.cpp %.h mi_arithmetic_utils.o
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

#Arithmetic util
mi_arithmetic_utils.o: mi_arithmetic_utils.cpp mi_arithmetic_utils.h $(DOUBLE_ARITHMETIC_OBJS) $(SEE_ARITHMETIC_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_arithmetic_utils.cpp $(ADDITINAL_INCLUDE_DIR)

#Hit test
mi_intersection_test.o: mi_intersection_test.cpp mi_intersection_test.h mi_arithmetic_utils.o $(SHAPE_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_intersection_test.cpp $(ADDITINAL_INCLUDE_DIR)

#RSA
mi_rsa_utils.o: mi_rsa_utils.h mi_rsa_utils.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_rsa_utils.cpp $(ADDITINAL_INCLUDE_DIR)

#Volume data statistic
mi_volume_statistician.o: mi_volume_statistician.h mi_volume_statistician.inl mi_volume_statistician.cpp mi_arithmetic_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_volume_statistician.cpp $(ADDITINAL_INCLUDE_DIR)

#RLE encode/decode
mi_run_length_operator.o: mi_run_length_operator.cpp mi_run_length_operator.h
	$(CC) -c -fPIC $(CFLAGS) mi_run_length_operator.cpp $(ADDITINAL_INCLUDE_DIR)

#Connected domain calculation
mi_connected_domain_analysis.o: mi_connected_domain_analysis.h mi_connected_domain_analysis.cpp mi_arithmetic_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_connected_domain_analysis.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
