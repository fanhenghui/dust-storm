#ifndef MED_IMAGING_ARITHMETIC_COLOR_UNIT_H_
#define MED_IMAGING_ARITHMETIC_COLOR_UNIT_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export RGBUnit
{
public:
    unsigned char r,g,b;

public:
    RGBUnit():r(0),g(0),b(0)
    {}

    explicit RGBUnit(unsigned char rr , unsigned char gg , unsigned char bb):r(rr),g(gg),b(bb)
    {
    };

    explicit RGBUnit(float fRNorm , float fGNorm, float fBNorm)
    {
        fRNorm*= 255.0f;
        fGNorm*= 255.0f;
        fBNorm*= 255.0f;

        fRNorm = fRNorm>255.0f ? 255.0f : fRNorm;
        fRNorm = fRNorm < 0.0f ? 0.0f : fRNorm;

        fGNorm = fGNorm>255.0f ? 255.0f : fGNorm;
        fGNorm = fGNorm < 0.0f ? 0.0f : fGNorm;

        fBNorm = fBNorm>255.0f ? 255.0f : fBNorm;
        fBNorm = fBNorm < 0.0f ? 0.0f : fBNorm;

        r = (unsigned char)fRNorm;
        g = (unsigned char)fGNorm;
        b = (unsigned char)fBNorm;
    }

    ~RGBUnit() {};

    void Print()
    {
        std::cout << "( " << (int)r << " , " << (int)g << " , " << (int)b << " )";
    }

private:
};

class Arithmetic_Export RGBAUnit
{
public:
    unsigned char r,g,b,a;

public:
    explicit RGBAUnit():r(0),g(0),b(0),a(0)
    {}

    explicit RGBAUnit(unsigned char rr , unsigned char gg , unsigned char bb):r(rr),g(gg),b(bb),a(255)
    {
    };

    explicit RGBAUnit(float fRNorm , float fGNorm, float fBNorm)
    {
        fRNorm*= 255.0f;
        fGNorm*= 255.0f;
        fBNorm*= 255.0f;

        fRNorm = fRNorm>255.0f ? 255.0f : fRNorm;
        fRNorm = fRNorm < 0.0f ? 0.0f : fRNorm;

        fGNorm = fGNorm>255.0f ? 255.0f : fGNorm;
        fGNorm = fGNorm < 0.0f ? 0.0f : fGNorm;

        fBNorm = fBNorm>255.0f ? 255.0f : fBNorm;
        fBNorm = fBNorm < 0.0f ? 0.0f : fBNorm;

        r = (unsigned char)fRNorm;
        g = (unsigned char)fGNorm;
        b = (unsigned char)fBNorm;
        a = 255;
    }

    explicit RGBAUnit(unsigned char rr , unsigned char gg , unsigned char bb , unsigned char aa):r(rr),g(gg),b(bb),a(aa)
    {
    };

    explicit RGBAUnit(float fRNorm , float fGNorm, float fBNorm , float fANorm)
    {
        fRNorm*= 255.0f;
        fGNorm*= 255.0f;
        fBNorm*= 255.0f;

        fRNorm = fRNorm>255.0f ? 255.0f : fRNorm;
        fRNorm = fRNorm < 0.0f ? 0.0f : fRNorm;

        fGNorm = fGNorm>255.0f ? 255.0f : fGNorm;
        fGNorm = fGNorm < 0.0f ? 0.0f : fGNorm;

        fBNorm = fBNorm>255.0f ? 255.0f : fBNorm;
        fBNorm = fBNorm < 0.0f ? 0.0f : fBNorm;

        fANorm = fANorm>255.0f ? 255.0f : fANorm;
        fANorm = fANorm < 0.0f ? 0.0f : fANorm;

        r = (unsigned char)fRNorm;
        g = (unsigned char)fGNorm;
        b = (unsigned char)fBNorm;
        a = (unsigned char)fANorm;
    }

    ~RGBAUnit() {};

    void Print()
    {
        std::cout << "( " << (int)r << " , " << (int)g << " , " << (int)b << " , " << (int)a<< " )";
    }

private:
};


MED_IMAGING_END_NAMESPACE

#endif