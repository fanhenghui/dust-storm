ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEF = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEF)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/glresource
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

GL_OBJS = mi_gl_environment.o mi_gl_context.o mi_gl_program.o mi_gl_buffer.o\
mi_gl_texture_1d_array.o mi_gl_texture_1d.o mi_gl_texture_2d.o \
mi_gl_texture_3d.o mi_gl_fbo.o \
mi_gl_vao.o mi_gl_time_query.o mi_gl_texture_cache.o mi_gl_utils.o \

OBJS = $(GL_OBJS) mi_gl_resource_manager_container.o 
TARGET = $(BIN_DIR)/libmedimg_glresource.so

EXPORT_HEADER = mi_gl_resource_export.h mi_gl_object.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log_setup -lGL -lGLEW -lmedimg_log -lmedimg_util 

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_log.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_gl_resource_manager_container.o: mi_gl_resource_manager_container.cpp mi_gl_resource_manager_container.h $(GL_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_gl_resource_manager_container.cpp $(ADDITINAL_INCLUDE_DIR)
	
$(GL_OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)	

.PHONY : clean
clean:
	-rm *.o $(TARGET)
