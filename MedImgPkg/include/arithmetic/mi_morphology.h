#ifndef MEDIMGARITHMETIC_MI_MORPHOLOGY_H
#define MEDIMGARITHMETIC_MI_MORPHOLOGY_H

#include "arithmetic/mi_arithmetic_export.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class Morphology
{
public:
    void dilate(unsigned char* mask, int width , int height , int loop) {
        std::unique_ptr<unsigned char[]> cur_mask(new unsigned char[width*height]);
        memcpy(cur_mask.get(), mask, width*height);

        unsigned char* mask0 = mask;
        unsigned char* mask1 = cur_mask.get();

        bool exchange = false;
        for (int i = 0 ; i<loop ; ++i) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = y*width + x;
                    if (mask1[idx] != 0) {
                        fill_9_n(mask0, width, height, x, y, mask1[idx]);
                    }
                }
            }

            if (i+1>=loop) {
                break;
            }

            unsigned char* mask_tmp = mask0;
            mask0 = mask1;
            mask1 = mask_tmp;
            exchange = !exchange;
        }

        if (exchange) {
            memcpy(mask, cur_mask.get(), width*height);
        }
    }

    void dilate(unsigned char* mask, int width , int height , int depth , int loop) {
        std::unique_ptr<unsigned char[]> cur_mask(new unsigned char[width*height*depth]);
        memcpy(cur_mask.get(), mask, width*height*depth);

        unsigned char* mask0 = mask;
        unsigned char* mask1 = cur_mask.get();

        bool exchange = false;
        for (int i = 0 ; i<loop ; ++i) {
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        int idx = z*width*height + y*width + x;
                        if (mask1[idx] != 0) {
                            fill_27_n(mask0, width, height, depth , x, y, z , mask1[idx]);
                        }
                    }
                }
            }

            if (i+1>=loop) {
                break;
            }

            unsigned char* mask_tmp = mask0;
            mask0 = mask1;
            mask1 = mask_tmp;
            exchange = !exchange;
        }

        if (exchange) {
            memcpy(mask, cur_mask.get(), width*height);
        }
    }

    void erose(unsigned char* mask, int width, int height, int loop) {
        std::unique_ptr<unsigned char[]> cur_mask(new unsigned char[width*height]);
        memcpy(cur_mask.get(), mask, width*height);

        unsigned char* mask0 = mask;
        unsigned char* mask1 = cur_mask.get();

        bool exchange = false;
        for (int i = 0; i < loop; ++i) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = y*width + x;
                    if (mask1[idx] == 0) {
                        fill_9_n(mask0, width, height, x, y, 0);
                    }
                }
            }

            if (i + 1 >= loop) {
                break;
            }

            unsigned char* mask_tmp = mask0;
            mask0 = mask1;
            mask1 = mask_tmp;
            exchange = !exchange;
        }

        if (exchange) {
            memcpy(mask, cur_mask.get(), width*height);
        }
    }

    void erose(unsigned char* mask, int width, int height, int depth ,int loop) {
        std::unique_ptr<unsigned char[]> cur_mask(new unsigned char[width*height*depth]);
        memcpy(cur_mask.get(), mask, width*height*depth);

        unsigned char* mask0 = mask;
        unsigned char* mask1 = cur_mask.get();

        bool exchange = false;
        for (int i = 0; i < loop; ++i) {
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        int idx = z*width*height + y*width + x;
                        if (mask1[idx] == 0) {
                            fill_27_n(mask0, width, height, depth , x, y, z, 0);
                        }
                    }
                }
            }
            

            if (i + 1 >= loop) {
                break;
            }

            unsigned char* mask_tmp = mask0;
            mask0 = mask1;
            mask1 = mask_tmp;
            exchange = !exchange;
        }

        if (exchange) {
            memcpy(mask, cur_mask.get(), width*height);
        }
    }

    void fill_9_n(unsigned char* mask, int width, int height, int x, int y , unsigned char label)
    {
        const int yyy = y + 2 > height ? height : y + 2;
        const int xxx = x + 2 > width ? width : x + 2;
        for (int yy = y - 1 < 0 ? 0 : y - 1; yy < yyy; ++yy) {
            for (int xx = x - 1 < 0 ? 0 : x - 1; xx < xxx; ++xx) {
                int idx = yy*width + xx;
                mask[idx] = label;
            }
        }
    }

    void fill_27_n(unsigned char* mask, int width, int height, int depth , int x, int y , int z , unsigned char label) {
        const int zzz = z + 2 > depth ? depth : z + 2;
        const int yyy = y + 2 > height ? height : y + 2;
        const int xxx = x + 2 > width ? width : x + 2;
        for (int zz = z - 1 < 0 ? 0 : z - 1; zz < zzz; ++zz) {
            for (int yy = y - 1 < 0 ? 0 : y - 1; yy < yyy; ++yy) {
                for (int xx = x - 1 < 0 ? 0 : x - 1; xx < xxx; ++xx) {
                    int idx = zz*width*height + yy*width + xx;
                    mask[idx] = label;
                }
            }
        }
        
    }
protected:
private:
};

MED_IMG_END_NAMESPACE
#endif
