CC = g++ -Wall

main run print_msg: run.cpp main.cpp print_msg.cpp
	$(CC) -o main main.cpp
	$(CC) -o run run.cpp
	$(CC) -o print_msg print_msg.cpp	

.PHONY : clean
clean :
	rm main run print_msg