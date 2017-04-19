extern void UT_CPUMPREntryExitPoints(int argc , char* argv[]);
extern void UT_CPUMPR(int argc , char* argv[]);
extern void UT_MeshRendering(int argc , char* argv[]);
extern void UT_BrickPool(int argc , char* argv[]);
extern void UT_CompureShader(int argc , char* argv[]);
extern void UT_GPUMPR(int argc , char* argv[]);

void main(int argc , char* argv[])
{
    UT_GPUMPR(argc , argv);
}