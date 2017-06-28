CC = g++ -std=c++11 -Wall
TARGET = be
$(TARGET) : be.cpp common.h
	$(CC)  -o be be.cpp -lpthread
.PHONY : clean
clean :
	rm $(TARGET)