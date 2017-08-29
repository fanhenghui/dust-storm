ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/arithmetic
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_point2.o mi_point3.o mi_vector2.o mi_vector3.o mi_matrix4.o \
mi_camera_base.o mi_ortho_camera.o mi_perspective_camera.o mi_track_ball.o \
mi_vector2f.o mi_vector3f.o mi_vector4f.o mi_matrix4f.o \
mi_arithmetic_utils.o \
mi_aabb.o mi_ellipsoid.o mi_line_segment.o mi_line.o mi_plane.o mi_triangle.o mi_intersection_test.o \
mi_rsa_utils.o mi_volume_statistician.o

TARGET = $(BIN_DIR)/libmedimg_arithmetic.so

EXPORT_HEADER = mi_arithmetic_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lmbedcrypto -lmbedtls -lmbedx509 

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_volume_statistician.o: $(EXPORT_HEADER) mi_volume_statistician.h mi_volume_statistician.cpp \
mi_ellipsoid.o mi_arithmetic_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_volume_statistician.cpp $(ADDITINAL_INCLUDE_DIR) 

mi_rsa_utils.o: $(EXPORT_HEADER) mi_rsa_utils.h mi_rsa_utils.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_rsa_utils.cpp $(ADDITINAL_INCLUDE_DIR) 

mi_intersection_test.o: $(EXPORT_HEADER) mi_intersection_test.h mi_intersection_test.cpp \
mi_aabb.o mi_ellipsoid.o mi_line_segment.o mi_line.o mi_plane.o 
	$(CC) -c -fPIC $(CFLAGS) mi_intersection_test.cpp $(ADDITINAL_INCLUDE_DIR) 

mi_arithmetic_utils.o: $(EXPORT_HEADER) mi_arithmetic_utils.h mi_arithmetic_utils.cpp \
mi_point3.o mi_point2.o mi_vector3.o mi_vector2.o mi_matrix4.o \
mi_vector2f.o mi_vector3f.o mi_vector4f.o  mi_matrix4f.o \
mi_ellipsoid.o
	$(CC) -c -fPIC $(CFLAGS) mi_arithmetic_utils.cpp $(ADDITINAL_INCLUDE_DIR)
	
mi_aabb.o: $(EXPORT_HEADER) mi_aabb.h mi_aabb.cpp mi_shape_interface.h mi_point3.o
	$(CC) -c -fPIC $(CFLAGS) mi_aabb.cpp $(ADDITINAL_INCLUDE_DIR) 

mi_ellipsoid.o: $(EXPORT_HEADER) mi_ellipsoid.h mi_ellipsoid.cpp mi_shape_interface.h \
mi_point3.o mi_vector3.o
	$(CC) -c -fPIC $(CFLAGS) mi_ellipsoid.cpp $(ADDITINAL_INCLUDE_DIR)

mi_line_segment.o: $(EXPORT_HEADER) mi_line_segment.h mi_line_segment.cpp mi_shape_interface.h \
mi_point3.o mi_point2.o mi_vector3.o mi_vector2.o
	$(CC) -c -fPIC $(CFLAGS) mi_line_segment.cpp $(ADDITINAL_INCLUDE_DIR)

mi_line.o: $(EXPORT_HEADER) mi_line.h mi_line.cpp mi_shape_interface.h \
mi_point3.o mi_point2.o
	$(CC) -c -fPIC $(CFLAGS) mi_line.cpp $(ADDITINAL_INCLUDE_DIR)

mi_plane.o: $(EXPORT_HEADER) mi_plane.h mi_plane.cpp mi_shape_interface.h \
mi_point3.o mi_point2.o
	$(CC) -c -fPIC $(CFLAGS) mi_plane.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_triangle.o: $(EXPORT_HEADER) mi_triangle.h mi_triangle.cpp mi_shape_interface.h \
mi_point3.o mi_point2.o
	$(CC) -c -fPIC $(CFLAGS) mi_triangle.cpp $(ADDITINAL_INCLUDE_DIR)		

mi_matrix4f.o: $(EXPORT_HEADER) mi_matrix4f.h mi_matrix4f.cpp mi_simd.h mi_vector3f.o mi_vector4f.o
	$(CC) -c -fPIC $(CFLAGS) mi_matrix4f.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector4f.o: $(EXPORT_HEADER) mi_vector4f.h mi_vector4f.cpp mi_simd.h mi_vector3f.o
	$(CC) -c -fPIC $(CFLAGS) mi_vector4f.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector3f.o: $(EXPORT_HEADER) mi_vector3f.h mi_vector3f.cpp mi_simd.h
	$(CC) -c -fPIC $(CFLAGS) mi_vector3f.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vector2f.o: $(EXPORT_HEADER) mi_vector2f.h mi_vector2f.cpp mi_simd.h
	$(CC) -c -fPIC $(CFLAGS) mi_vector2f.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_track_ball.o: $(EXPORT_HEADER) mi_track_ball.h mi_track_ball.cpp \
mi_quat4.h mi_point3.o mi_point2.o mi_vector3.o mi_vector2.o mi_matrix4.o
	$(CC) -c -fPIC $(CFLAGS) mi_track_ball.cpp $(ADDITINAL_INCLUDE_DIR)

mi_perspective_camera.o: $(EXPORT_HEADER) mi_perspective_camera.h mi_perspective_camera.cpp mi_camera_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_perspective_camera.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ortho_camera.o: $(EXPORT_HEADER) mi_ortho_camera.h mi_ortho_camera.cpp mi_camera_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_ortho_camera.cpp $(ADDITINAL_INCLUDE_DIR)

mi_camera_base.o: $(EXPORT_HEADER) mi_camera_base.h mi_camera_base.cpp \
mi_quat4.h mi_point3.o mi_point2.o mi_vector3.o mi_vector2.o mi_matrix4.o
	$(CC) -c -fPIC $(CFLAGS) mi_camera_base.cpp $(ADDITINAL_INCLUDE_DIR)

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

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
