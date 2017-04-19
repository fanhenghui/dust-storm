#ifndef ARITHMETIC_VECTOR2D_H_
#define ARITHMETIC_VECTOR2D_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export Vector2
{
public:
    double x, y;
    static const Vector2 kZeroVector;

public:
    ~Vector2()
    {
    }

    Vector2() :x(0), y(0)
    {
    }

    Vector2(double x1, double y1): x(x1), y(y1)
    {
    }

    Vector2(const Vector2& v)
    {
        x = v.x;
        y = v.y;
    }

    inline Vector2& operator += (const Vector2& v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }

    inline Vector2 operator + (const Vector2& v) const
    {
        return Vector2(x + v.x, y + v.y);
    }

    inline Vector2& operator -= (const Vector2& v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    inline Vector2 operator - (const Vector2& v) const
    {
        return Vector2(x - v.x, y - v.y);
    }

    inline Vector2 operator - () const
    {
        return Vector2(-x, -y);
    }

    inline Vector2& operator *= (double scale)
    {
        x *= scale;
        y *= scale;
        return *this;
    }

    inline Vector2 operator * (double scale) const
    {
        return Vector2(x * scale, y * scale);
    }

    inline bool operator != (const Vector2& v) const
    {
        return (std::fabs(x - v.x) > DOUBLE_EPSILON ||
            std::fabs(y - v.y) > DOUBLE_EPSILON);
    }

    inline bool operator == (const Vector2& v) const
    {
        return (std::fabs(x - v.x) < DOUBLE_EPSILON &&
            std::fabs(y - v.y) < DOUBLE_EPSILON);
    }

    inline Vector2& operator = (const Vector2& v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }

    inline double AngleBetween(const Vector2& v) const
    {
        double len = Magnitude() * v.Magnitude();

        len = (len > DOUBLE_EPSILON) ? len : DOUBLE_EPSILON;

        double dot = DotProduct(v) / len;
        dot = (std::min)(dot, 1.0);
        dot = (std::max)(dot, -1.0);
        return std::acos(dot);
    }

    inline double DotProduct(const Vector2& v) const
    {
        return x * v.x + y * v.y;
    }

    inline double Magnitude() const
    {
        return std::sqrt(x * x + y * y);
    }

    inline void Normalize()
    {
        double len = std::sqrt(x * x + y * y);
        if (len < DOUBLE_EPSILON)
        {
            ARITHMETIC_THROW_EXCEPTION("Vector's magnitude is 0 ! Get normalize failed!");
        }
        double leninv = 1.0 / len;
        x *= leninv;
        y *= leninv;
    }

    Vector2 GetNormalize() const
    {
        double len = std::sqrt(x * x + y * y);
        if (len < DOUBLE_EPSILON)
        {
            ARITHMETIC_THROW_EXCEPTION("Vector's magnitude is 0 ! Get normalize failed!");
        }
        double leninv = 1.0 / len;
        return Vector2(x * leninv, y * leninv);
    }

    bool Parallel(const Vector2& v) const
    {
        return std::fabs(this->DotProduct(v) - this->Magnitude()*v.Magnitude()) < DOUBLE_EPSILON;
    }

    bool Orthogonal(const Vector2& v) const
    {
        return std::fabs(this->DotProduct(v)) < DOUBLE_EPSILON;
    }

    void Print()
    {
        std::cout <<"( "<<x << " , "<<y << " ) ";
    }
};

Vector2 Arithmetic_Export operator *(double scale, const Vector2& v);

double Arithmetic_Export AngleBetween(const Vector2& v1, const Vector2& v2);

Vector2 Arithmetic_Export CrossProduct(const Vector2& v1, const Vector2& v2);

double Arithmetic_Export DotProduct(const Vector2& v1, const Vector2& v2);

bool Arithmetic_Export Parallel(const Vector2& v1, const Vector2& v2);

bool Arithmetic_Export Orthogonal(const Vector2& v1, const Vector2& v2);

MED_IMAGING_END_NAMESPACE

#endif