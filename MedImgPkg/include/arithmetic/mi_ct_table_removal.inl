template<typename T>
void convert_16_to_8(T* data,  const unsigned int (&dim)[3] , 
                    float ww , float wl , unsigned char* data_8) { 
    const float wl_min = wl - ww*0.5f;
    for (int i = 0 ; i < dim[0]*dim[1]*dim[2] ; ++i)
    {
        float v = (float)(data[i]);
        v = (v - wl_min)/ww;
        v = v > 1.0f ? 1.0f : v;
        v = v < 0.0f ? 0.0f : v;
        data_8[i] = (unsigned char)(v*255.0f);
    }
}

template <class T>
void CTTableRemoval<T>::remove() {
    clock_t _time0,_time1;
    clock_t _start;
    _time0 = clock();
    _start = _time0;

    const unsigned int volume_size = _dim[0]*_dim[1]*_dim[2];
    const unsigned int img_size = _dim[0]*_dim[1];
    const int w = _dim[0];
    const int h = _dim[1];

    //1 convert to 8bit
    const float WW_BONE = 1500;
    const float WL_BONE = 300;
    float ww = WW_BONE/_slope;
    float wl = WL_BONE - _intercept;
    unsigned char* data_8 = new unsigned char[volume_size];
    convert_16_to_8(_data_ref, _dim, ww, wl, data_8);

    _time1 = clock();
    MI_ARITHMETIC_LOG(MI_DEBUG) << "convert to 8 cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
    _time0 = _time1;
    
    //2 smmoth filter (basic gaussian)
    std::unique_ptr<unsigned char[]> data_8_img(new unsigned char[w*h]);
    for (int z = 0 ; z < _dim[2] ; ++z) {
        unsigned char* cur_data = data_8 + z*w*h;
        unsigned char v00,v01,v02,v10,v11,v12,v20,v21,v22;

        for (int y = 1; y< h-1; ++y) {
            for (int x = 1; x < w-1; ++x) {
                v00 = cur_data[(y-1)*w + (x-1)];
                v01 = cur_data[(y-1)*w + (x)];
                v02 = cur_data[(y-1)*w + (x+1)];

                v10 = cur_data[(y)*w + (x-1)];
                v11 = cur_data[(y)*w + (x)];
                v12 = cur_data[(y)*w + (x+1)];

                v20 = cur_data[(y+1)*w + (x-1)];
                v21 = cur_data[(y+1)*w + (x)];
                v22 = cur_data[(y+1)*w + (x+1)];
            
                float v = v00*1.0f + v01*2.0f + v02*1.0f + 
                          v10*2.0f + v11*4.0f + v12*2.0f +
                          v20*1.0f + v21*2.0f + v22*1.0f;
                v/= 16.0f;
                data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
            }
        }
        //4 border
        for (int x = 1; x < w-1; ++x) {
            int y = 0;
            v10 = cur_data[(y)*w + (x-1)];
            v11 = cur_data[(y)*w + (x)];
            v12 = cur_data[(y)*w + (x+1)];

            v20 = cur_data[(y+1)*w + (x-1)];
            v21 = cur_data[(y+1)*w + (x)];
            v22 = cur_data[(y+1)*w + (x+1)];

            v00 = v11;
            v01 = v11;
            v02 = v11;

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        for (int x = 1; x < w-1; ++x) {
            int y = h-1;
            v00 = cur_data[(y-1)*w + (x-1)];
            v01 = cur_data[(y-1)*w + (x)];
            v02 = cur_data[(y-1)*w + (x+1)];

            v10 = cur_data[(y)*w + (x-1)];
            v11 = cur_data[(y)*w + (x)];
            v12 = cur_data[(y)*w + (x+1)];

            v20 = v11;
            v21 = v11;
            v22 = v11;

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        for (int y = 1; y < h-1; ++y) {            
            int x = 0;
            v01 = cur_data[(y-1)*w + (x)];
            v02 = cur_data[(y-1)*w + (x+1)];
            
            v11 = cur_data[(y)*w + (x)];
            v12 = cur_data[(y)*w + (x+1)];

            v21 = cur_data[(y+1)*w + (x)];
            v22 = cur_data[(y+1)*w + (x+1)];

            v00 = v11;
            v10 = v11;
            v20 = v11;

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        for (int y = 1; y < h-1; ++y) {   
            int x = w-1;
            v00 = cur_data[(y-1)*w + (x-1)];
            v01 = cur_data[(y-1)*w + (x)];

            v10 = cur_data[(y)*w + (x-1)];
            v11 = cur_data[(y)*w + (x)];

            v20 = cur_data[(y+1)*w + (x-1)];
            v21 = cur_data[(y+1)*w + (x)];

            v02 = v11;
            v12 = v11;
            v22 = v11;

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        //four corner
        {
            int x = 0;
            int y = 0;
            v11 = cur_data[(y)*w + (x)];

            v00 = v11;
            v01 = v11;
            v02 = v11;

            v10 = v11;
            v12 = cur_data[(y)*w + (x+1)];;

            v20 = v11;
            v21 = cur_data[(y+1)*w + (x)];
            v22 = cur_data[(y+1)*w + (x+1)];

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        {
            int x = 0;
            int y = h-1;
            v11 = cur_data[(y)*w + (x)];

            v00 = v11;
            v01 = cur_data[(y-1)*w + (x)];
            v02 = cur_data[(y-1)*w + (x+1)];
    
            v10 = v11;
            v12 = cur_data[(y)*w + (x+1)];
    
            v20 = v11;
            v21 = v11;
            v22 = v11;
            

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        {
            int x = w-1;
            int y = 0;
            v11 = cur_data[(y)*w + (x)];

            v00 = v11;
            v01 = v11;
            v02 = v11;

            v10 = cur_data[(y)*w + (x-1)];
            v12 = v11;

            v20 = cur_data[(y+1)*w + (x-1)];
            v21 = cur_data[(y+1)*w + (x)];
            v22 = v11;
            

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        {
            int x = w-1;
            int y = h-1;
            v11 = cur_data[(y)*w + (x)];

            v00 = cur_data[(y-1)*w + (x-1)];
            v01 = cur_data[(y-1)*w + (x)];
            v02 = v11;

            v10 = cur_data[(y)*w + (x-1)];
            v11 = cur_data[(y)*w + (x)];
            v12 = v11;

            v20 = v11;
            v21 = v11;
            v22 = v11;

            float v = v00*1.0f + v01*2.0f + v02*1.0f + 
            v10*2.0f + v11*4.0f + v12*2.0f +
            v20*1.0f + v21*2.0f + v22*1.0f;

            v/= 16.0f;
            data_8_img.get()[(y)*w + (x)] = (unsigned char)v;
        }

        memcpy(cur_data, data_8_img.get(), w*h);
    }

    _time1 = clock();
    MI_ARITHMETIC_LOG(MI_DEBUG) << "smooth filter cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
    _time0 = _time1;

    FileUtil::write_raw("/home/wangrui22/data/test_filter.raw", data_8, _dim[0]*_dim[1]*_dim[2]);

    //3 threshold segment
    float th = (_body_threshold - wl + ww*0.5f)/ww;
    th = th < 0 ? 0:th;
    th = th > 1 ? 1:th;
    unsigned char th_8 = (unsigned char)(th*255.0f);
    MI_ARITHMETIC_LOG(MI_DEBUG) << "body threshold in 8: " << (int)th_8; 
    for (int z = 0 ; z < _dim[2] ; ++z) {
        const int w = _dim[0];
        const int h = _dim[1];
        unsigned char* cur_data = data_8 + z*w*h;
        SegmentThreshold<unsigned char> seg;
        unsigned int dim2[3] = {w,h,1};
        seg.set_dim(dim2);
        seg.set_data_ref(cur_data);
        seg.set_mask_ref(_mask_ref + z*w*h);
        seg.set_target_label(_target_label);
        seg.set_min_scalar(0);
        seg.set_max_scalar(255);
        seg.segment(th_8);
    }

    FileUtil::write_raw("/home/wangrui22/data/test_th_seg.raw", _mask_ref, _dim[0]*_dim[1]*_dim[2]);
    _time1 = clock();
    MI_ARITHMETIC_LOG(MI_DEBUG) << "threshold segment cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
    _time0 = _time1;

    //4 region growing background
    unsigned char* mask_rg = data_8;
    memset(mask_rg, _target_label, volume_size);
    std::stack<unsigned int> seeds;//add 8 corner seeds
    unsigned int s000 = 0;
    unsigned int s001 = (_dim[2]-1) * (_dim[0]*_dim[1]);
    unsigned int s010 = (_dim[1]-1) * _dim[0];
    unsigned int s011 = (_dim[1]-1) * _dim[0] + (_dim[2]-1) * (_dim[0]*_dim[1]);
    unsigned int s100 = _dim[0]-1;
    unsigned int s101 = _dim[0]-1 + (_dim[2]-1) * (_dim[0]*_dim[1]);
    unsigned int s110 = _dim[0]-1 + (_dim[1]-1) * _dim[0];
    unsigned int s111 = _dim[0]-1 + (_dim[1]-1) * _dim[0] + (_dim[2]-1) * (_dim[0]*_dim[1]);
    unsigned int s8[8] = {s000,s001,s010,s011,s100,s101,s110,s111};
    for (int i = 0; i < 8; ++i) {
        seeds.push(s8[i]);
        mask_rg[s8[i]] = 0;
    } 
    while(!seeds.empty()) {
        unsigned int idx = seeds.top();
        seeds.pop();
        int z = idx / (img_size);
        int y = (idx - z*img_size)/_dim[0];
        int x = idx - z*img_size - y*_dim[0];

        //get 27 neighbour
        const int zzz = z + 2 > _dim[2] ? _dim[2] : z + 2;
        const int yyy = y + 2 > _dim[1] ? _dim[1] : y + 2;
        const int xxx = x + 2 > _dim[0] ? _dim[0] : x + 2;
        for (unsigned int zz = z - 1 < 0 ? 0 : z - 1 ; zz < zzz ; ++zz) {
            for (unsigned int yy = y - 1 < 0 ? 0 : y - 1; yy < yyy ; ++yy) {
                for (unsigned int xx = x - 1 < 0 ? 0 : x - 1; xx < xxx ; ++xx) {
                    if (xx != x || yy != y || zz != z) {
                        unsigned int idx2 = zz * _dim[0] * _dim[1] + yy * _dim[0] + xx;
                        if (_mask_ref[idx2] == 0 && mask_rg[idx2] != 0) {
                            mask_rg[idx] = 0;
                            seeds.push(idx2);
                        }
                    }
                }
            }
        }
    }

    _time1 = clock();
    MI_ARITHMETIC_LOG(MI_DEBUG) << "region grow background cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
    _time0 = _time1;
    FileUtil::write_raw("/home/wangrui22/data/test_rg.raw", mask_rg, _dim[0]*_dim[1]*_dim[2]);


    //way 1 remove table
    const int way = 0;
    if (way == 1) {
        //5 transform to sagittal
        const int axial_type = check_axial();
        if (axial_type == -1) {
            //not axial, don't support now
            memcpy(_mask_ref, mask_rg, volume_size);
            delete [] mask_rg;
            return;
        } 
        unsigned char* mask_sag = new unsigned char[volume_size];
        to_sagittal(mask_rg, mask_sag, axial_type);
        delete [] mask_rg;
        mask_rg = nullptr;

        _time1 = clock();
        MI_ARITHMETIC_LOG(MI_DEBUG) << "convert to sigttal cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
        _time0 = _time1;

        FileUtil::write_raw("/home/wangrui22/data/test_sag.raw", mask_sag, _dim[0]*_dim[1]*_dim[2]);
        
        //6 remove bed region
        const unsigned int dim_s[3] = {_dim[1], _dim[2], _dim[0]};
        std::unique_ptr<unsigned char[]> data_8_img_s(new unsigned char[dim_s[0]*dim_s[1]]);
        
        //6.1 find bed seed (TODO use hough transform to check line seed)
        const int line_pixel_p = int(0.7*dim_s[1]);
        const int search_step = 2;
        int search_x_begin = dim_s[0] - 1 - 5+ search_step;
        int search_x_end = dim_s[0] -1 + search_step;
        std::stack<unsigned int> line_seeds;
        do {
            search_x_begin -= search_step;
            search_x_end -= search_step;
            if (search_x_end < 0){
                break;
            }
            for (unsigned int z = 0; z < dim_s[2]; ++z) {
                unsigned char* cur_mask = mask_sag + z*dim_s[0]*dim_s[1];
                std::stack<unsigned int> cur_seed;
                int gap = 0;
                for (unsigned int x = search_x_begin; x < search_x_end ; ++x) {
                    for (unsigned int y = 0; y < dim_s[1]; ++y) {
                        if (cur_mask[y*dim_s[0] + x] != 0) {
                            if (gap == 0) {
                                gap = 1;
                            }
                            cur_seed.push(y*dim_s[0] + x);
                        } else {
                            if (gap == 1) {
                                if (cur_seed.size() > line_pixel_p) {
                                    while(!cur_seed.empty()) {
                                        cur_mask[cur_seed.top()] = 0;
                                        line_seeds.push(cur_seed.top() + z*dim_s[0]*dim_s[1]);
                                        cur_seed.pop();
                                    }
                                }
                                gap = 0;
                            }
                        }
                    }
                    if (cur_seed.size() > line_pixel_p) {
                        while(!cur_seed.empty()) {
                            cur_mask[cur_seed.top()] = 0;
                            line_seeds.push(cur_seed.top() + z*dim_s[0]*dim_s[1]);
                            cur_seed.pop();
                        }
                    }
                }        
            }
        }while (line_seeds.empty());

        //6.2 region grow to remove table
        while(!line_seeds.empty()) {
            unsigned int idx = line_seeds.top();
            line_seeds.pop();
            int z = idx / (img_size);
            int y = (idx - z*img_size)/_dim[0];
            int x = idx - z*img_size - y*_dim[0];

            //get 27 neighbour
            const int zzz = z + 2 > _dim[2] ? _dim[2] : z + 2;
            const int yyy = y + 2 > _dim[1] ? _dim[1] : y + 2;
            const int xxx = x + 2 > _dim[0] ? _dim[0] : x + 2;
            for (unsigned int zz = z - 1 < 0 ? 0 : z - 1 ; zz < zzz ; ++zz) {
                for (unsigned int yy = y - 1 < 0 ? 0 : y - 1; yy < yyy ; ++yy) {
                    for (unsigned int xx = x - 1 < 0 ? 0 : x - 1; xx < xxx ; ++xx) {
                        if (xx != x || yy != y || zz != z) {
                            unsigned int idx2 = zz * _dim[0] * _dim[1] + yy * _dim[0] + xx;
                            if (mask_sag[idx2] == _target_label) {
                                mask_sag[idx2] = 0;
                                line_seeds.push(idx2);
                            }
                        }
                    }
                }
            }
        }
        FileUtil::write_raw("/home/wangrui22/data/test_sag_removal_table.raw", mask_sag, _dim[0]*_dim[1]*_dim[2]);

        _time1 = clock();
        MI_ARITHMETIC_LOG(MI_DEBUG) << "remove table in sigttal cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
        _time0 = _time1;

        //7 transform sagittal to axial
        back_t0_original(_mask_ref, mask_sag, axial_type);
        delete [] mask_sag;
        mask_sag = nullptr;

        _time1 = clock();
        MI_ARITHMETIC_LOG(MI_DEBUG) << "transform sagittal back to axial cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
        _time0 = _time1;

        //8 dialet
        Morphology mor;
        mor.dilate(_mask_ref, _dim[0], _dim[1], _dim[2], 2);

        _time1 = clock();
        MI_ARITHMETIC_LOG(MI_DEBUG) << "dialet cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
        _time0 = _time1;
        

        MI_ARITHMETIC_LOG(MI_INFO) << "ct table removal cost: " << (double)(_time1 - _start)/CLOCKS_PER_SEC;
    }

    if (way == 0) {
        //in volume center to find seed in rect
        int rect_size = 1;
        int rect_step = 3;
        int c_x = _dim[0]/2;
        int c_y = _dim[1]/2;
        int c_z = _dim[2]/2;
        std::stack<unsigned int> body_seeds;
        do {
            rect_size += rect_step;
            int x_begin = c_x - rect_size;
            int x_end = c_x + rect_size;
            int y_begin = c_y - rect_size;
            int y_end = c_y + rect_size;
            int z_begin = c_z - rect_size;
            int z_end = c_z + rect_size;
            if (x_begin < 0 || y_begin < 0 || z_begin < 0 ||
                x_end > _dim[0]-1 || y_end > _dim[1]-1 || z_end > _dim[2]-1) {
                break;
            }

            unsigned int idx;
            for (unsigned int z = z_begin; z < z_end; ++z) {
                for (unsigned int y = y_begin; y < y_end; ++y) {
                    for (unsigned int x = x_begin; x < x_end; ++x) {
                        idx = z*_dim[0]*_dim[1] + y*_dim[0] + x;
                        if (_mask_ref[idx] == _target_label) {
                            _mask_ref[idx] = _target_label + 1;
                            body_seeds.push(idx);
                        }
                    }
                }
            }
        
        }while(body_seeds.empty());

        memcpy(_mask_ref, mask_rg, volume_size);
        delete [] mask_rg;

        if (body_seeds.empty()) {
            return;
        }

        while(!body_seeds.empty()) {
            unsigned int idx = body_seeds.top();
            body_seeds.pop();
            int z = idx / (img_size);
            int y = (idx - z*img_size)/_dim[0];
            int x = idx - z*img_size - y*_dim[0];

            //get 27 neighbour
            const int zzz = z + 2 > _dim[2] ? _dim[2] : z + 2;
            const int yyy = y + 2 > _dim[1] ? _dim[1] : y + 2;
            const int xxx = x + 2 > _dim[0] ? _dim[0] : x + 2;
            for (unsigned int zz = z - 1 < 0 ? 0 : z - 1 ; zz < zzz ; ++zz) {
                for (unsigned int yy = y - 1 < 0 ? 0 : y - 1; yy < yyy ; ++yy) {
                    for (unsigned int xx = x - 1 < 0 ? 0 : x - 1; xx < xxx ; ++xx) {
                        if (xx != x || yy != y || zz != z) {
                            unsigned int idx2 = zz * _dim[0] * _dim[1] + yy * _dim[0] + xx;
                            if (_mask_ref[idx2] == _target_label) {
                                _mask_ref[idx2] = _target_label + 1;
                                body_seeds.push(idx2);
                            }
                        }
                    }
                }
            }
        }

        for (unsigned int i = 0; i < volume_size; ++i) {
            if (_mask_ref[i] != _target_label + 1) {
                _mask_ref[i] = 0;
            } else {
                _mask_ref[i] = 1;
            }
        }


        _time1 = clock();
        MI_ARITHMETIC_LOG(MI_DEBUG) << "region grow body part cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
        _time0 = _time1;

        //8 dialet
        Morphology mor;
        mor.dilate(_mask_ref, _dim[0], _dim[1], _dim[2], 2);

        _time1 = clock();
        MI_ARITHMETIC_LOG(MI_DEBUG) << "dialet cost: " << (double)(_time1 - _time0)/CLOCKS_PER_SEC;
        _time0 = _time1;
        
        MI_ARITHMETIC_LOG(MI_INFO) << "ct table removal cost: " << (double)(_time1 - _start)/CLOCKS_PER_SEC;

    }
    
}

// scan_type
//0 axial(+z)
//1 axial(-z)
//-1 not axial
template<class T>
int CTTableRemoval<T>::check_axial() {
    if (fabs(_image_orientation[0].x) > fabs(_image_orientation[0].y) &&
        fabs(_image_orientation[0].x) > fabs(_image_orientation[0].z) &&
        fabs(_image_orientation[1].y) > fabs(_image_orientation[1].x) &&
        fabs(_image_orientation[1].y) > fabs(_image_orientation[1].z)) {
        return _image_orientation[2].z > 0 ? 0 : 1;
    } else {
        return -1;
    }
}

template<class T>
void CTTableRemoval<T>::to_sagittal(unsigned char* axial_mask, unsigned char* sagittal_mask, int ori_axial_type) {
    const unsigned int dim_s[3] = {_dim[1], _dim[2], _dim[0]};
    unsigned int idx_axial = 0;
    unsigned int idx_sagittal = 0;
    unsigned int xx,yy,zz;
    if (0 == ori_axial_type) { // sagittal x = axial -y ; sagittal y = axial -z ; sagittal z = axial x
        for (unsigned int z = 0; z < _dim[2]; ++z) {
            for (unsigned int y = 0; y < _dim[1]; ++y) {
                for (unsigned int x = 0; x < _dim[0]; ++x) {
                    idx_axial = z*_dim[0]*_dim[1] + y*_dim[0] + x;
                    zz = x;
                    yy = (_dim[2] - 1 - z);
                    xx = y;
                    idx_sagittal = zz*dim_s[1]*dim_s[0] + yy*dim_s[0] + xx;
                    sagittal_mask[idx_sagittal] = axial_mask[idx_axial];
                }   
            }
        }
    } else {
        for (unsigned int z = 0; z < _dim[2]; ++z) { // sagittal x = axial -y ; sagittal y = axial z ; sagittal z = axial x
            for (unsigned int y = 0; y < _dim[1]; ++y) {
                for (unsigned int x = 0; x < _dim[0]; ++x) {
                    idx_axial = z*_dim[0]*_dim[1] + y*_dim[0] + x;
                    zz = x;
                    yy = z;
                    xx = y;
                    idx_sagittal = zz*dim_s[1]*dim_s[0] + yy*dim_s[0] + xx;
                    sagittal_mask[idx_sagittal] = axial_mask[idx_axial];
                }   
            }
        }
    }
}

template<class T>
void CTTableRemoval<T>::back_t0_original(unsigned char* axial_mask, unsigned char* sagittal_mask, int ori_axial_type) {
    const unsigned int dim_s[3] = {_dim[1], _dim[2], _dim[0]};
    unsigned int idx_axial = 0;
    unsigned int idx_sagittal = 0;
    unsigned int xx,yy,zz;
    if (0 == ori_axial_type) { // sagittal x = axial -y ; sagittal y = axial -z ; sagittal z = axial x
        for (unsigned int z = 0; z < _dim[2]; ++z) {
            for (unsigned int y = 0; y < _dim[1]; ++y) {
                for (unsigned int x = 0; x < _dim[0]; ++x) {
                    idx_axial = z*_dim[0]*_dim[1] + y*_dim[0] + x;
                    zz = x;
                    yy = (_dim[2] - 1 - z);
                    xx = y;
                    idx_sagittal = zz*dim_s[1]*dim_s[0] + yy*dim_s[0] + xx;
                    axial_mask[idx_axial] = sagittal_mask[idx_sagittal];
                }   
            }
        }
    } else {
        for (unsigned int z = 0; z < _dim[2]; ++z) { // sagittal x = axial -y ; sagittal y = axial z ; sagittal z = axial x
            for (unsigned int y = 0; y < _dim[1]; ++y) {
                for (unsigned int x = 0; x < _dim[0]; ++x) {
                    idx_axial = z*_dim[0]*_dim[1] + y*_dim[0] + x;
                    zz = x;
                    yy = z;
                    xx = y;
                    idx_sagittal = zz*dim_s[1]*dim_s[0] + yy*dim_s[0] + xx;
                    axial_mask[idx_axial] = sagittal_mask[idx_sagittal];
                }   
            }
        }
    }
}