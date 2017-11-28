// Copyright 2014 Baidu Inc. All Rights Reserved.
// Author: xinglongliu (liuxinglong01@baidu.com)
//
// med interface from cpp to python module

#ifndef AI_LUNG_EVAULATE_PY_WRAPPER_H
#define AI_LUNG_EVAULATE_PY_WRAPPER_H

#include <vector>
#include <memory>

#include "python2.7/Python.h"

namespace medical_ai{

// constants
#define MAX_PATH 4096

class AILungEvaulatePyWrapper
{
public:
    /*
    * return value for test new images
    */
    struct SPredictedNodule {
        char  sUID[MAX_PATH];
        float fX, fY, fZ;
        float fRadius;
        float fProb;
    };
    typedef std::vector<SPredictedNodule> VPredictedNodules;

public:
    AILungEvaulatePyWrapper() : _module(NULL) {}

    ~AILungEvaulatePyWrapper() {
         clean_module();
    }

    /* more (non-static) functions here */
    int init(const char* py_home, const char* interface_path);

    /**
     *
     * @param str_full_inputpath
     * @param buffer pointer to the target value, should not alloced or freed out of the method scope
     * @param length of corresponding buffer
     * @return
     */
    int preprocess(const char *str_full_inputpath, char* &buffer, int& length);


    /**
     *
     * @param str_full_inputpath FULL path to input image, WITH .npz
     * @return packed SPredictedNodule structures in vector
     */
    int evaluate(const char *str_full_inputpath, VPredictedNodules &v_nodules);

    const char* get_last_err() const;
    const char* get_version();

private:
    void clean_module();
    PyObject* get_func_inmodule(const char *strFunName);
    int finalize_pyinterface();
    int initilize_interface();
    void set_err(const char* str);

private:
    PyObject* _module;

    //error info
    char _last_err[4096];
};

}


#endif //INTERFACE_CONFIG_H
