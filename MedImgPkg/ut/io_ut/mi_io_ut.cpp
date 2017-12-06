extern int pacs_ut(int argc, char* argv[]);
extern int targa_ut(int argc, char* argv[]);
extern int dicom_loader_ut(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    return pacs_ut(argc,argv);
}