#include <stdlib.h>

extern int TestMessageQueue(int argc, char* argv[]);
extern int TestLogger(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    return TestLogger(argc, argv);
}