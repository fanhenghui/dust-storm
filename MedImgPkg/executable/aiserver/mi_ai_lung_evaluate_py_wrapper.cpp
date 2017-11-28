#include "mi_ai_lung_evaluate_py_wrapper.h"

namespace medical_ai{

//
// env settings
//
const char* INTERFACE_MODULE                          = "interface";
// working methods
const char* PREPROCESS_NEW_IMG_TOBUFFER_METHOD        = "cpp_preprocess_dicomdir_tobuffer";
const char* EVALUATE_NEW_IMG_METHOD_TOVEC             = "cpp_evaluate_newimage_tovec";
// helper methods
const char* FINALIZE_INTERFACE                        = "finalize_interface";
const char* INITIALIZE_INTERFACE                      = "initialize_interface";
const char* GET_MODULE_VERSION                        = "get_module_version";


int AILungEvaulatePyWrapper::init(const char* py_home, const char* interface_path) {
    //
    // setting envs
    //
    char scpy[MAX_PATH];
    strncpy(scpy, py_home, MAX_PATH);
    Py_SetPythonHome(scpy);

    // char* python_home = Py_GetPythonHome();
    // printf("python home is %s\n", python_home);

    //
    // doing python work
    //
    Py_Initialize();

    // ugly, but no other choices since we are using python27
    // Py_SetPath is only provided for python3
    // should be used as:
    //    char python_path[1024];
    //    char* orig_path = Py_GetPath();
    //    int nRet = snprintf(python_path, 1024, "%s:%s", orig_path, LUNA_INTERFACE_PATH);
    //    Py_SetPath(python_path);
    char cCommand[MAX_PATH];
    int nRet = snprintf(cCommand, MAX_PATH, "sys.path.append('%s')", interface_path);
    if (nRet >= MAX_PATH) { 
        set_err("cCommand out of index");
        return -1;
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(cCommand);

    //
    // now, we are all setting down
    //
    _module = PyImport_ImportModule(INTERFACE_MODULE);
    if (!_module) {
        set_err("import python module test failed.");
        return -1;
    }

    // init interface
    initilize_interface();

    return  1;
}

PyObject* AILungEvaulatePyWrapper::get_func_inmodule(const char *strFunName)
{
    PyObject* pFunc = PyObject_GetAttrString(_module, strFunName);
    if (!pFunc)
    {
        set_err("import python func failed.");
        return NULL;
    }

    return pFunc;
}

void AILungEvaulatePyWrapper::clean_module()
{
    finalize_pyinterface();

    PyImport_Cleanup();
    Py_Finalize();
}

int AILungEvaulatePyWrapper::initilize_interface()
{
    int nRet = -1;
    PyObject* pFunc = get_func_inmodule(INITIALIZE_INTERFACE);
    if (pFunc != NULL)
    {
        PyObject* pResult = PyObject_CallObject(pFunc, NULL);
        if (pResult)
        {
            PyArg_Parse(pResult, "i", &nRet);
            Py_DECREF(pResult);
        }
    }

    return  nRet;
}

int AILungEvaulatePyWrapper::finalize_pyinterface() {
    int nRet = -1;
    PyObject* pFunc = get_func_inmodule(FINALIZE_INTERFACE);
    if (pFunc != NULL) {
        PyObject* pResult = PyObject_CallObject(pFunc, NULL);
        if (pResult) {
            PyArg_Parse(pResult, "i", &nRet);
            Py_DECREF(pResult);
        }
    }
    return  nRet;
}


int AILungEvaulatePyWrapper::preprocess(const char *str_fullinputpath, char *&buffer, int& length) {
    int nRet = -1;
    buffer = NULL;
    length = 0;

    PyObject* pFunc = get_func_inmodule(PREPROCESS_NEW_IMG_TOBUFFER_METHOD);
    if (pFunc != NULL) {
        PyObject* pParam = Py_BuildValue("(s)", str_fullinputpath);
        PyObject* pResult = PyObject_CallObject(pFunc, pParam);
        if (pResult) {
            // PyTuple_GetItem only borrows item from list/turple, not increasing ref count
            // so we should NOT dec ref
            char* pTmpBuffer = NULL;
            PyObject* pRET = PyTuple_GetItem(pResult, 0);
            PyObject* pBuffer = PyTuple_GetItem(pResult, 1);
            PyArg_Parse(pRET, "n", &nRet);
            PyArg_Parse(pBuffer, "z#", &pTmpBuffer, &length);

            buffer = (char*)malloc(sizeof(char)* length);
            memcpy(buffer, pTmpBuffer, sizeof(char) * length);

            Py_DECREF(pResult);
        }
        Py_DECREF(pParam);
    }
    return  nRet;
}

int AILungEvaulatePyWrapper::evaluate(const char *str_full_inputpath, VPredictedNodules &v_nodules) {
    v_nodules.clear();

    PyObject* pFunc = get_func_inmodule(EVALUATE_NEW_IMG_METHOD_TOVEC);
    if (pFunc != NULL)
    {
        PyObject *pParam = Py_BuildValue("(s)", str_full_inputpath);
        PyObject *pResult = PyObject_CallObject(pFunc, pParam);
        if (pResult) {
            int szList = PyList_Size(pResult);
            PyObject *item;
            for (int i = 0; i < szList; ++i) {
                PyObject *pRet = PyList_GetItem(pResult, i);
                SPredictedNodule sNodule;
                if (PyList_Check(pRet)) {
                    item = PyList_GetItem(pRet, 0);

                    // string objects are different in getting the return values
                    // 'z' represents the  returned tmpstr could be NULL !
                    char *tmpstr = NULL;
                    PyArg_Parse(item, "z", &tmpstr);
                    if (tmpstr != NULL)
                        strncpy(sNodule.sUID, tmpstr, MAX_PATH);
                    else
                        strncpy(sNodule.sUID, "NON_VALID_SUID", MAX_PATH);

                    item = PyList_GetItem(pRet, 1);
                    PyArg_Parse(item, "f", &sNodule.fX);
                    item = PyList_GetItem(pRet, 2);
                    PyArg_Parse(item, "f", &sNodule.fY);
                    item = PyList_GetItem(pRet, 3);
                    PyArg_Parse(item, "f", &sNodule.fZ);
                    item = PyList_GetItem(pRet, 4);
                    PyArg_Parse(item, "f", &sNodule.fRadius);
                    item = PyList_GetItem(pRet, 5);
                    PyArg_Parse(item, "f", &sNodule.fProb);
                    v_nodules.push_back(sNodule);
                }
            }
            Py_DECREF(pResult);
        }
        Py_DECREF(pParam);
    }

    if (v_nodules.empty())
        return -1;
    else
        return  1;
}

void AILungEvaulatePyWrapper::set_err(const char* str) {
    memset(_last_err, 0, sizeof(_last_err));
    strcpy(_last_err, str);
}

const char* AILungEvaulatePyWrapper::get_last_err() const {
    return _last_err;
}


const char* AILungEvaulatePyWrapper::get_version() {
    char *str_version = NULL;
    PyObject* pFunc = get_func_inmodule(GET_MODULE_VERSION);
    if (pFunc != NULL)
    {
        PyObject* pResult = PyObject_CallObject(pFunc, NULL);
        if (pResult)
        {
            PyArg_Parse(pResult, "s", &str_version);
            // Py_DECREF(pResult);
        }
    }
    return  str_version;
}


}








