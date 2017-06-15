#Core makefile

#version (default debug)
ver = debug

#compiler
CC = g++ -std=c++11

#file direction
MODULE_NAME = MultiResolutionImageInterface
INCLUDE_DIR = ../../Include/IO
LIB_DIR = ../../Lib

VPATH = $(INCLUDE_DIR) : $(INCLUDE_DIR)/$(MODULE_NAME)
#vpath %.h $(INCLUDE_DIR):$(INCLUDE_DIR)/$(MODULE_NAME)

#debug/release cc
ifeq ($(ver) , debug)
CFLAGS += -c -g -fPIC
else
CFLAGS += -c -fPIC
endif

#macro
# DEFS = OPENSLIDEFILEFORMAT_STATIC_DEFINE 
# CFLAGS += -D $(DEFS)

#target and obj
TARGET = $(LIB_DIR)/libmultiresolutionimageinterface.so
OBJS = JPEG2000Codec.o TileCache.o AperioSVSWriter.o \
MultiResolutionImage.o MultiResolutionImageFactory.o MultiResolutionImageReader.o MultiResolutionImageWriter.o \
LIFImage.o LIFImageFactory.o \
TIFFImage.o TIFFImageFactory.o \
VSIImage.o VSIImageFactory.o VSIImageReader.o


EXPORT_HEADER = multiresolutionimageinterface_export.h
$(TARGET): $(OBJS) $(EXPORT_HEADER)
	$(CC) -shared -fpic -lijg8 -ltiff -llibjasper -llibjpeg -lzlib -lpugixml -lboost_system -lboost_filesystem -lboost_regex -lboost_thread \
	-lboost_chrono -lboost_date_time -lboost_atomic -lmbedcrypto -lmbedtls -lmbedx509 -o $(TARGET) $(OBJS)

MultiResolutionImageReader.o : $(EXPORT_HEADER) MultiResolutionImageReader.cpp MultiResolutionImageReader.h MultiResolutionImageFactory.o
	$(CC) $(CFLAGS) MultiResolutionImageReader.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

MultiResolutionImageFactory.o : $(EXPORT_HEADER) MultiResolutionImageFactory.cpp MultiResolutionImageFactory.h VSIImage.o MultiResolutionImage.o 
	$(CC) $(CFLAGS) MultiResolutionImageFactory.cpp -I$(INCLUDE_DIR)  -I$(INCLUDE_DIR)/$(MODULE_NAME)

MultiResolutionImageWriter.o : $(EXPORT_HEADER) MultiResolutionImageWriter.cpp MultiResolutionImageWriter.h MultiResolutionImage.o JPEG2000Codec.o
	$(CC) $(CFLAGS) MultiResolutionImageWriter.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)	

VSIImageReader.o: $(EXPORT_HEADER) VSIImageReader.cpp VSIImageReader.h JPEG2000Codec.o
	$(CC) $(CFLAGS) VSIImageReader.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

VSIImageFactory.o : $(EXPORT_HEADER) VSIImageFactory.cpp VSIImageFactory.h VSIImage.o 
	$(CC) $(CFLAGS) LIFImageFactory.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

VSIImage.o : $(EXPORT_HEADER) VSIImage.cpp VSIImage.h MultiResolutionImage.o
	$(CC) $(CFLAGS) VSIImage.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

LIFImageFactory.o : $(EXPORT_HEADER) LIFImageFactory.cpp LIFImageFactory.h LIFImage.o 
	$(CC) $(CFLAGS) LIFImageFactory.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

LIFImage.o : $(EXPORT_HEADER) LIFImage.cpp LIFImage.h MultiResolutionImage.o
	$(CC) $(CFLAGS) LIFImage.cpp -I$(INCLUDE_DIR)  -I$(INCLUDE_DIR)/$(MODULE_NAME)

TIFFImage.o : $(EXPORT_HEADER) TIFFImage.cpp TIFFImage.h MultiResolutionImage.o
	$(CC) $(CFLAGS) TIFFImage.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

TIFFImageFactory.o : $(EXPORT_HEADER) TIFFImageFactory.cpp TIFFImageFactory.h TIFFImage.o 
	$(CC) $(CFLAGS) TIFFImageFactory.cpp -I$(INCLUDE_DIR)/$(MODULE_NAME)

MultiResolutionImage.o : $(EXPORT_HEADER) MultiResolutionImage.cpp MultiResolutionImage.h $(LIB_DIR)/Core.so
	$(CC) $(CFLAGS) MultiResolutionImage.cpp -I$(INCLUDE_DIR)  -I$(INCLUDE_DIR)/$(MODULE_NAME)

AperioSVSWriter.o : $(EXPORT_HEADER) AperioSVSWriter.cpp AperioSVSWriter.h JPEG2000Codec.o
	$(CC) $(CFLAGS) AperioSVSWriter.cpp -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/$(MODULE_NAME)

TileCache.o : $(EXPORT_HEADER) TileCache.cpp TileCache.h
	$(CC) $(CFLAGS) TileCache.cpp -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/$(MODULE_NAME)

JPEG2000Codec.o : $(EXPORT_HEADER) JPEG2000Codec.cpp JPEG2000Codec.h
	$(CC) $(CFLAGS) JPEG2000Codec.cpp -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/$(MODULE_NAME)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)