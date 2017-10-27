#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <stack>

#include "util/mi_file_util.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "arithmetic/mi_connected_domain_analysis.h"
#include "arithmetic/mi_segment_threshold.h"
#include "arithmetic/mi_morphology.h"

using namespace medical_imaging;

std::shared_ptr<ImageDataHeader> _data_header;
std::shared_ptr<ImageData> _volume_data;

template<typename T>
int convert_volume_8(
    T* data, 
    const unsigned int (&dim)[3] ,
    float intercept , float slope , 
    float ww , float wl ,
    unsigned char* data_8) { 
   
    const float wl_min = wl - ww*0.5f;
#ifndef _DEBUG
#pragma omp parallel  for
#endif
    for (int i = 0 ; i < dim[0]*dim[1]*dim[2] ; ++i)
    {
        float v = (float)(data[i]);
        v = (v + intercept)/slope;
        v = (v - wl_min)/ww;
        v = v > 1.0f ? 1.0f : v;
        v = v < 0.0f ? 0.0f : v;
        data_8[i] = (unsigned char)(v*255.0f);
    }

    return 0;
}


int main(int argc , char* argv[]) {

    const std::string root = "/home/wangrui22/data/demo/lungs2/orig/1.3.6.1.4.1.14519.5.2.1.6279.6001.188059920088313909273628445208/";
    std::vector<std::string> files;
    std::set<std::string> dcm_postfix;
    dcm_postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(root, dcm_postfix, files);
    DICOMLoader loader;
    IOStatus status = loader.load_series(files, _volume_data, _data_header);
    if(status != IO_SUCCESS) {
        printf("load file root %s failed.\n" , root.c_str());
        return -1;
    }
    printf("load DICOM %s done.\n" , _data_header->series_uid.c_str());
    printf("dim : %u %u %u \n" , _volume_data->_dim[0] , _volume_data->_dim[0] , _volume_data->_dim[2]);
    const unsigned int dim[3] = {_volume_data->_dim[0] , _volume_data->_dim[1] , _volume_data->_dim[2]};
    unsigned char* data_8 = new unsigned char[dim[0]*dim[1]*dim[2]];
    unsigned char* mask = new unsigned char[dim[0]*dim[1]*dim[2]];
    
    const float WW_BONE = 1500;
    const float WL_BONE = 300;
    const float WW_ADBOMEN  = 400;
    const float WL_ADBOMEN= 60;
    float ww = WW_BONE;
    float wl = WL_BONE;
    if (_volume_data->_data_type == USHORT) {
        unsigned short* data = (unsigned short*)_volume_data->get_pixel_pointer();
        convert_volume_8(data , dim , _volume_data->_intercept , _volume_data->_slope , ww , wl , data_8);
    }else {
        short* data = (short*)_volume_data->get_pixel_pointer();
        convert_volume_8(data , dim , _volume_data->_intercept , _volume_data->_slope , ww , wl , data_8);
    }

    FileUtil::write_raw("/home/wangrui22/data/demo/lungs2/data_8.raw" , data_8 , dim[0]*dim[1]*dim[2]);
    printf("16->8 done.\n");

    memset(mask , 0 , dim[0]*dim[1]*dim[2]);
#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (int z = 0 ; z < dim[2] ; ++z) {
        unsigned char* cur_data = data_8 + z*dim[0]*dim[1];
        SegmentThreshold<unsigned char> seg;
        unsigned int dim2[3] = {dim[0] ,dim[1] , 1};
        seg.set_dim(dim2);
        seg.set_data_ref(cur_data);
        seg.set_mask_ref(mask + z*dim[0]*dim[1]);
        seg.set_target_label(1);
        seg.set_min_scalar(0);
        seg.set_max_scalar(255);
        float th = seg.get_threshold_otsu_i();
        seg.segment((unsigned char)th);
    }

    printf("segmentation done.\n");
    FileUtil::write_raw("/home/wangrui22/data/demo/lungs2/mask.raw" , mask , dim[0]*dim[1]*dim[2]);

    Morphology mor;
    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);

    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);

    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);

    printf("morphology done.\n");
    FileUtil::write_raw("/home/wangrui22/data/demo/lungs2/mask_mor.raw" , mask , dim[0]*dim[1]*dim[2]);

    ConnectedDomainAnalysis conn;
    conn.set_dim(dim);
    conn.set_mask_ref(mask);
    conn.set_target_label(1);
    unsigned int begin[3] ={0,0,0};
    conn.set_roi(begin, dim,  mask);
    conn.keep_major();

    printf("keep major connected domain done.\n");
    FileUtil::write_raw("/home/wangrui22/data/demo/lungs2/mask_major.raw" , mask , dim[0]*dim[1]*dim[2]);


    mor.dilate(mask, dim[0], dim[1], dim[2], 1);
    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);
    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);
    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);
    mor.erose(mask, dim[0], dim[1], dim[2], 1);
    mor.dilate(mask, dim[0], dim[1], dim[2], 1);
    mor.erose(mask, dim[0], dim[1], dim[2], 1);

    printf("morphology major domain done.\n");
    FileUtil::write_raw("/home/wangrui22/data/demo/lungs2/mask_major_mor.raw" , mask , dim[0]*dim[1]*dim[2]);

    //std::fstream in("D:/temp/mask_major_mor.raw" , std::ios::binary | std::ios::in);
    //in.read((char*)mask , dim[0]*dim[1]*dim[2]);
    //in.close();

    //scan fill
    unsigned char* mask_lr = new unsigned char[dim[0]*dim[1]*dim[2]];
    memcpy(mask_lr , mask , dim[0]*dim[1]*dim[2]);
    unsigned char* mask_ud = new unsigned char[dim[0]*dim[1]*dim[2]];
    memcpy(mask_ud , mask , dim[0]*dim[1]*dim[2]);
    for (int z = 0; z < dim[2]; ++z) {
        for (int y = 0; y < dim[1]; ++y) {
            std::stack<int> x_fill;
            unsigned char head_label = mask[z*dim[0]*dim[1] + y*dim[0] + 0];
            //unsigned char last_label = mask[z*dim[0]*dim[1] + y*dim[0] + 0];
            if (head_label == 1) {
                x_fill.push(0);
            }
            for (int x = 1 ; x < dim[0] ; ++x) {
                if (0 == head_label) {
                    if ( mask[z*dim[0]*dim[1] + y*dim[0] + x] == 1) {
                        x_fill.push(x);
                        head_label = 1;
                    }
                }else if (1 == head_label) {
                    if ( mask[z*dim[0]*dim[1] + y*dim[0] + x] == 0) {
                        x_fill.push(x);
                    } else {
                        if (x_fill.size() > 1) {
                            //printf("scan : %i pixel on z(%i) y(%i). \n" , int(x_fill.size()) , z , y);
                            while(!x_fill.empty()) {
                                int cur_x = x_fill.top();
                                x_fill.pop();
                                mask_lr[z*dim[0]*dim[1] + y*dim[0] + cur_x] = 1;
                            }
                            x_fill.push(x);
                        } else {
                            x_fill.pop();
                            x_fill.push(x);
                        }
                    }
                }
            }
        }
    }

    for (int z = 0; z < dim[2]; ++z) {
        for (int x = 0; x < dim[0]; ++x) {
            std::stack<int> y_fill;
            unsigned char head_label = mask[z*dim[0]*dim[1] + x];
            //unsigned char last_label = mask[z*dim[0]*dim[1] + y*dim[0] + 0];
            if (head_label == 1) {
                y_fill.push(0);
            }
            for (int y = 1 ; y < dim[1] ; ++y) {
                if (0 == head_label) {
                    if ( mask[z*dim[0]*dim[1] + y*dim[0] + x] == 1) {
                        y_fill.push(y);
                        head_label = 1;
                    }
                }else if (1 == head_label) {
                    if ( mask[z*dim[0]*dim[1] + y*dim[0] + x] == 0) {
                        y_fill.push(y);
                    } else {
                        if (y_fill.size() > 1) {
                            //printf("scan : %i pixel on z(%i) y(%i). \n" , int(x_fill.size()) , z , y);
                            while(!y_fill.empty()) {
                                int cur_y = y_fill.top();
                                y_fill.pop();
                                mask_ud[z*dim[0]*dim[1] + cur_y*dim[0] + x] = 1;
                            }
                            y_fill.push(y);
                        } else {
                            y_fill.pop();
                            y_fill.push(y);
                        }
                    }
                }
            }
        }
    }

    for (unsigned int i = 0 ;i<dim[0]*dim[1]*dim[2] ;++i)
    {
        if (mask_lr[i] == 1 && mask_ud[i] == 1)
        {
            mask[i] = 1;
        }
    }

    printf("scan to fill done.\n");
    FileUtil::write_raw("/home/wangrui22/data/demo/lungs2/mask_major_mor_fill.raw" , mask , dim[0]*dim[1]*dim[2]);

    return 0;
}