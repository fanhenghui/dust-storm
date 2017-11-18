extern int pacs_ut(int argc, char* argv[]);
extern int targa_ut(int argc, char* argv[]);
extern int dicom_loader_ut(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    return dicom_loader_ut(argc,argv);
}