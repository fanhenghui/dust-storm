CC = g++ -std=c++11 -Wall
TARGET = be fe
$(TARGET) : be.cpp fe.cpp common.h
	$(CC)  -o be be.cpp -lpthread
	$(CC)  -o fe fe.cpp -lpthread

.PHONY : clean
clean :
	rm $(TARGET)