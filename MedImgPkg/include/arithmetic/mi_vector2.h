#ifndef MEDIMGARITHMETIC_MI_VECTOR2_H
#define MEDIMGARITHMETIC_MI_VECTOR2_H

#include <ostream>
#include "arithmetic/mi_arithmetic_export.h"
#include "log/mi_logger_util.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Vector2 {
public:
    double x, y;
    static const Vector2 S_ZERO_VECTOR;

public:
    ~Vector2() {}

    Vector2() : x(0), y(0) {}

    Vector2(double x1, double y1) : x(x1), y(y1) {}

    Vector2(const Vector2& v) {
        x = v.x;
        y = v.y;
    }

    inline Vector2& operator+=(const Vector2& v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    inline Vector2 operator+(const Vector2& v) const {
        return Vector2(x + v.x, y + v.y);
    }

    inline Vector2& operator-=(const Vector2& v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    inline Vector2 operator-(const Vector2& v) const {
        return Vector2(x - v.x, y - v.y);
    }

    inline Vector2 operator-() const {
        return Vector2(-x, -y);
    }

    inline Vector2& operator*=(double scale) {
        x *= scale;
        y *= scale;
        return *this;
    }

    inline Vector2 operator*(double scale) const {
        return Vector2(x * scale, y * scale);
    }

    inline bool operator!=(const Vector2& v) const {
        return (std::fabs(x - v.x) > DOUBLE_EPSILON ||
                std::fabs(y - v.y) > DOUBLE_EPSILON);
    }

    inline bool operator==(const Vector2& v) const {
        return (std::fabs(x - v.x) < DOUBLE_EPSILON &&
                std::fabs(y - v.y) < DOUBLE_EPSILON);
    }

    inline Vector2& operator=(const Vector2& v) {
        x = v.x;
        y = v.y;
        return *this;
    }

    inline double angle_between(const Vector2& v) const {
        double len = magnitude() * v.magnitude();

        len = (len > DOUBLE_EPSILON) ? len : DOUBLE_EPSILON;

        double dot = dot_product(v) / len;
        dot = (std::min)(dot, 1.0);
        dot = (std::max)(dot, -1.0);
        return std::acos(dot);
    }

    inline double dot_product(const Vector2& v) const {
        return x * v.x + y * v.y;
    }

    inline double magnitude() const {
        return std::sqrt(x * x + y * y);
    }

    inline void normalize() {
        double len = std::sqrt(x * x + y * y);

        if (len < DOUBLE_EPSILON) {
            //LoggerUtil::log(MI_WARNING , "Arithmetic" , "vector3's magnitude is 0 ! Get normalize failed.");
            return;
        }

        double leninv = 1.0 / len;
        x *= leninv;
        y *= leninv;
    }

    Vector2 get_normalize() const {
        double len = std::sqrt(x * x + y * y);

        if (len < DOUBLE_EPSILON) {
            //LoggerUtil::log(MI_WARNING , "Arithmetic" , "vector3's magnitude is 0 ! Get normalize failed.");
            return *this;
        }

        double leninv = 1.0 / len;
        return Vector2(x * leninv, y * leninv);
    }

    bool parallel(const Vector2& v) const {
        return std::fabs(this->dot_product(v) - this->magnitude() * v.magnitude()) <
               DOUBLE_EPSILON;
    }

    bool orthogonal(const Vector2& v) const {
        return std::fabs(this->dot_product(v)) < DOUBLE_EPSILON;
    }

    friend std::ostream& operator<<(std::ostream &strm, const Vector2 &pt) {
        strm << "(" << pt.x << "," << pt.y << ") ";
        return strm;
    }
};

Vector2 Arithmetic_Export operator*(double scale, const Vector2& v);

double Arithmetic_Export angle_between(const Vector2& v1, const Vector2& v2);

Vector2 Arithmetic_Export cross(const Vector2& v1, const Vector2& v2);

double Arithmetic_Export dot_product(const Vector2& v1, const Vector2& v2);

bool Arithmetic_Export parallel(const Vector2& v1, const Vector2& v2);

bool Arithmetic_Export orthogonal(const Vector2& v1, const Vector2& v2);

MED_IMG_END_NAMESPACE

#endif