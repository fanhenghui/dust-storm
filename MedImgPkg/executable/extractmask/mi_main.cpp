#include <string>

extern int logic(int argc , char* argv[]);
extern int logic_vis(int argc , char* argv[]);

int main(int argc , char* argv[]) {
    bool vis = false;

    for (int i = 1; i < argc ; ++i) {
        if (std::string(argv[i]) == "-vis") {
            vis = true;
        }
    }

    if (vis) {
        return logic_vis(argc , argv);
    } else {
        return logic(argc , argv);
    }
}