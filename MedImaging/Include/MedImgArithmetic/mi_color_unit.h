#ifndef MED_IMAGING_ARITHMETIC_COLOR_UNIT_H_
#define MED_IMAGING_ARITHMETIC_COLOR_UNIT_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"
#include "MedImgArithmetic/mi_vector4f.h"

MED_IMAGING_BEGIN_NAMESPACE

struct RGBUnit
{
    unsigned char r,g,b;

    RGBUnit():r(0),g(0),b(0)
    {}

    explicit RGBUnit(unsigned char rr , unsigned char gg , unsigned char bb):r(rr),g(gg),b(bb)
    {
    };

    explicit RGBUnit(float r_norm , float g_norm, float b_norm)
    {
        r_norm*= 255.0f;
        g_norm*= 255.0f;
        b_norm*= 255.0f;

        r_norm = r_norm>255.0f ? 255.0f : r_norm;
        r_norm = r_norm < 0.0f ? 0.0f : r_norm;

        g_norm = g_norm>255.0f ? 255.0f : g_norm;
        g_norm = g_norm < 0.0f ? 0.0f : g_norm;

        b_norm = b_norm>255.0f ? 255.0f : b_norm;
        b_norm = b_norm < 0.0f ? 0.0f : b_norm;

        r = (unsigned char)r_norm;
        g = (unsigned char)g_norm;
        b = (unsigned char)b_norm;
    }

    bool operator == (const RGBUnit& rgb) const
    {
        return rgb.r == r && rgb.g ==g && rgb.b == b;
    }

    bool operator != (const RGBUnit& rgb) const
    {
        return rgb.r != r || rgb.g !=g || rgb.b != b;
    }

    void print()
    {
        std::cout << "( " << (int)r << " , " << (int)g << " , " << (int)b << " )";
    }
};

struct RGBAUnit
{
    unsigned char r,g,b,a;

    explicit RGBAUnit():r(0),g(0),b(0),a(0)
    {}

    explicit RGBAUnit(unsigned char rr , unsigned char gg , unsigned char bb):r(rr),g(gg),b(bb),a(255)
    {
    };

    explicit RGBAUnit(float r_norm , float g_norm, float b_norm)
    {
        r_norm*= 255.0f;
        g_norm*= 255.0f;
        b_norm*= 255.0f;

        r_norm = r_norm>255.0f ? 255.0f : r_norm;
        r_norm = r_norm < 0.0f ? 0.0f : r_norm;

        g_norm = g_norm>255.0f ? 255.0f : g_norm;
        g_norm = g_norm < 0.0f ? 0.0f : g_norm;

        b_norm = b_norm>255.0f ? 255.0f : b_norm;
        b_norm = b_norm < 0.0f ? 0.0f : b_norm;

        r = (unsigned char)r_norm;
        g = (unsigned char)g_norm;
        b = (unsigned char)b_norm;
        a = 255;
    }

    explicit RGBAUnit(unsigned char rr , unsigned char gg , unsigned char bb , unsigned char aa):r(rr),g(gg),b(bb),a(aa)
    {
    };

    explicit RGBAUnit(float r_norm , float g_norm, float b_norm , float a_norm)
    {
        r_norm*= 255.0f;
        g_norm*= 255.0f;
        b_norm*= 255.0f;

        r_norm = r_norm>255.0f ? 255.0f : r_norm;
        r_norm = r_norm < 0.0f ? 0.0f : r_norm;

        g_norm = g_norm>255.0f ? 255.0f : g_norm;
        g_norm = g_norm < 0.0f ? 0.0f : g_norm;

        b_norm = b_norm>255.0f ? 255.0f : b_norm;
        b_norm = b_norm < 0.0f ? 0.0f : b_norm;

        a_norm = a_norm>255.0f ? 255.0f : a_norm;
        a_norm = a_norm < 0.0f ? 0.0f : a_norm;

        r = (unsigned char)r_norm;
        g = (unsigned char)g_norm;
        b = (unsigned char)b_norm;
        a = (unsigned char)a_norm;
    }

    inline void blend(const RGBAUnit& top)
    {
        const float alpha = static_cast<float>(top.a)/255.0f;
        Vector4f v0((float)r ,(float)g , (float)b , (float)a);
        Vector4f v1((float)top.r ,(float)top.g , (float)top.b , (float)top.a);
        Vector4f v = v0*(1.0f-alpha) + v1*alpha;
        for (int i = 0 ; i< 4 ; ++i)
        {
            if (v._m[i] > 255.f)
            {
                v._m[i] = 255.0f;
            }
        }
        r = static_cast<unsigned char>(v[0]);
        g = static_cast<unsigned char>(v._m[1]);
        b = static_cast<unsigned char>(v._m[2]);
        a = static_cast<unsigned char>(v._m[3]);
    }

    bool operator == (const RGBAUnit& rgba) const
    {
        return rgba.r == r && rgba.g ==g && rgba.b == b && rgba.a == a;
    }

    bool operator != (const RGBAUnit& rgba) const
    {
        return rgba.r != r || rgba.g !=g || rgba.b != b || rgba.a != a;
    }

    void print()
    {
        std::cout << "( " << (int)r << " , " << (int)g << " , " << (int)b << " , " << (int)a<< " )";
    }
};

//bool Arithmetic_Export operator == (const RGBAUnit& left , const RGBAUnit& right);
//bool Arithmetic_Export operator == (const RGBUnit& left , const RGBUnit& right);


MED_IMAGING_END_NAMESPACE

#endif