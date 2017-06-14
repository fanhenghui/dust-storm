
make_file_test : linux_c_test.cpp class0.o class1.o
	g++ linux_c_test.cpp class0.o class1.o -lboost_system -lboost_thread -o make_file_test

class0.o : class0.cpp
	g++ -c class0.cpp

class1.o : class1.cpp
	g++ -c class1.cpp

clean :
	rm class0.o class1.o make_file_test