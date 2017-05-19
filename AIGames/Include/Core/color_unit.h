#ifndef MED_IMAGING_ARITHMETIC_COLOR_UNIT_H_
#define MED_IMAGING_ARITHMETIC_COLOR_UNIT_H_


struct RGBUnit
{
    unsigned char r,g,b;

    RGBUnit():r(0),g(0),b(0)
    {}

    explicit RGBUnit(unsigned char rr , unsigned char gg , unsigned char bb):r(rr),g(gg),b(bb)
    {
    };

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


    explicit RGBAUnit(unsigned char rr , unsigned char gg , unsigned char bb , unsigned char aa):r(rr),g(gg),b(bb),a(aa)
    {
    };

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




#endif