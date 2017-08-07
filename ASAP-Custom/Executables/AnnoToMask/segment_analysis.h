#ifndef SEGMNET_ANALYSIS_H
#define SEGMNET_ANALYSIS_H

#include <memory>
#include <limits>
#include <iostream>


class SegmentAnalysis
{
public:
    unsigned char get_threshold_otus_low(unsigned char* img_gray, int width , int height , AABB<int> aabb)
    {
        //get threshold
        const int gray_level = 256;
        unsigned int gray_hist[gray_level];
        memset(gray_hist, 0, sizeof(gray_hist));
        float sum = 0;

        int pixel_num = 0;
        for (int y = aabb._min[1] ; y < aabb._max[1] ; ++y)
        {
            for (int x = aabb._min[0] ; x < aabb._max[0] ; ++x)
            {
                unsigned char tmp = 255 - img_gray[y*width + x];
                gray_hist[tmp] += 1;
                sum += tmp * 0.001f;
                pixel_num += 1;
            }
        }

        //calculate mean
        const float pixel_num_f = (float)pixel_num;
        const float mean = sum / (float)pixel_num * 1000.0f;
        float max_icv = std::numeric_limits<float>::min();
        int max_gray_scalar = -1;
        for (int i = 1; i < gray_level - 1; ++i)
        {
            //PA MA
            unsigned int pixel_a = 0;
            float sum_a = 0.0f;
            for (int j = 0; j <= i; ++j)
            {
                pixel_a += gray_hist[j];
                sum_a += (float)j*gray_hist[j] * 0.001f;
            }
            float pa = (float)pixel_a / pixel_num_f;
            float mean_a = sum_a / (float)pixel_a * 1000.0f;

            //PB MB
            float pixel_b_f = (float)(pixel_num - pixel_a);
            float pb = pixel_b_f / pixel_num_f;
            float mean_b = (sum - sum_a) / pixel_b_f * 1000.0f;

            //ICV
            float icv = pa * (mean_a - mean)*(mean_a - mean) +
                pb * (mean_b - mean)*(mean_b - mean);

            if (icv > max_icv)
            {
                max_icv = icv;
                max_gray_scalar = i;
            }
        }   
        return (unsigned char)max_gray_scalar;
    }

    void segment_low(unsigned char* img_gray, unsigned char* mask, int width, int height , unsigned char th)
    {
        const int length = width*height;
        memset(mask, 0, sizeof(unsigned char)*length);
        for (int i = 0; i < length; ++i)
        {
            if (255 - img_gray[i] > th)
            {
                mask[i] = 1;
            }
        }
    }

protected:
private:
};

#endif