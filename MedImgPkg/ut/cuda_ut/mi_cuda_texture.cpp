extern "C"  int _cuda_texture(int argc, char* argv[]);

int cuda_texture(int argc, char* argv[]) {
    return _cuda_texture(argc, argv);
}