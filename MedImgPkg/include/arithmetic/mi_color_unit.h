#ifndef MEDIMGARITHMETIC_MI_COLOR_UNIT_H
#define MEDIMGARITHMETIC_MI_COLOR_UNIT_H

#include "arithmetic/mi_arithmetic_export.h"
#include <ostream>
#include "arithmetic/mi_vector4f.h"

MED_IMG_BEGIN_NAMESPACE

struct RGBUnit {
    unsigned char r, g, b;

    RGBUnit() : r(0), g(0), b(0) {
    }

    explicit RGBUnit(unsigned char rr, unsigned char gg, unsigned char bb) : r(rr), g(gg), b(bb) {};

    bool operator==(const RGBUnit& rgb) const {
        return rgb.r == r && rgb.g == g && rgb.b == b;
    }

    bool operator!=(const RGBUnit& rgb) const {
        return rgb.r != r || rgb.g != g || rgb.b != b;
    }

    friend std::ostream& operator << (std::ostream& strm , const RGBUnit& rgb) {
        strm << "( " << (int)rgb.r << " , " << (int)rgb.g << " , " << (int)rgb.b << " )";
        return strm;
    }
};

struct RGBAUnit {
    unsigned char r, g, b, a;

    explicit RGBAUnit() : r(0), g(0), b(0), a(0) {
    }

    explicit RGBAUnit(unsigned char rr, unsigned char gg, unsigned char bb) : r(rr), g(gg), b(bb),
        a(255) {};

    explicit RGBAUnit(unsigned char rr, unsigned char gg, unsigned char bb, unsigned char aa) : r(rr),
        g(gg), b(bb), a(aa) {};

    static RGBAUnit norm_to_integer(float r_norm, float g_norm, float b_norm) {
        r_norm *= 255.0f;
        g_norm *= 255.0f;
        b_norm *= 255.0f;

        r_norm = r_norm > 255.0f ? 255.0f : r_norm;
        r_norm = r_norm < 0.0f ? 0.0f : r_norm;

        g_norm = g_norm > 255.0f ? 255.0f : g_norm;
        g_norm = g_norm < 0.0f ? 0.0f : g_norm;

        b_norm = b_norm > 255.0f ? 255.0f : b_norm;
        b_norm = b_norm < 0.0f ? 0.0f : b_norm;

        return RGBAUnit((unsigned char)r_norm, (unsigned char)g_norm, (unsigned char)b_norm, 255);
    }

    inline void blend(const RGBAUnit& top) {
        const float alpha = static_cast<float>(top.a) / 255.0f;
        Vector4f v0((float)r, (float)g, (float)b, (float)a);
        Vector4f v1((float)top.r, (float)top.g, (float)top.b, (float)top.a);
        Vector4f v = v0 * (1.0f - alpha) + v1 * alpha;

        for (int i = 0; i < 4; ++i) {
            if (v._m[i] > 255.f) {
                v._m[i] = 255.0f;
            }
        }

        r = static_cast<unsigned char>(v[0]);
        g = static_cast<unsigned char>(v._m[1]);
        b = static_cast<unsigned char>(v._m[2]);
        a = static_cast<unsigned char>(v._m[3]);
    }

    bool operator==(const RGBAUnit& rgba) const {
        return rgba.r == r && rgba.g == g && rgba.b == b && rgba.a == a;
    }

    bool operator!=(const RGBAUnit& rgba) const {
        return rgba.r != r || rgba.g != g || rgba.b != b || rgba.a != a;
    }

    friend std::ostream& operator << (std::ostream& strm, RGBAUnit& rgba) {
        strm << "( " << (int)rgba.r << " , " << (int)rgba.g << " , " << (int)rgba.b << " , " << (int)rgba.a << " )";
        return strm;
    }
};

MED_IMG_END_NAMESPACE

#endif