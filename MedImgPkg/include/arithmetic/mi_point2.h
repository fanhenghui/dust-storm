#ifndef MEDIMGARITHMETIC_MI_POINT2_H
#define MEDIMGARITHMETIC_MI_POINT2_H

#include <ostream>
#include "arithmetic/mi_vector2.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Point2 {
public:
    double x, y;

    static const Point2 S_ZERO_POINT;

public:
    ~Point2() {}

    Point2() : x(0), y(0) {}

    Point2(double x1, double y1) : x(x1), y(y1) {}

    Point2(const Point2& pt) {
        x = pt.x;
        y = pt.y;
    }

    inline Point2& operator+=(const Vector2& v) {
        this->x = this->x + v.x;
        this->y = this->y + v.y;
        return *this;
    }

    inline Point2& operator-=(const Vector2& v) {
        this->x = this->x - v.x;
        this->y = this->y - v.y;
        return *this;
    }

    inline Point2& operator*=(double scale) {
        x *= scale;
        y *= scale;
        return *this;
    }

    inline Point2 operator*(double scale) const {
        return Point2(x * scale, y * scale);
    }

    inline Point2& operator=(const Point2& pt) {
        this->x = pt.x;
        this->y = pt.y;
        return *this;
    }

    inline bool operator!=(const Point2& pt) const {
        return (std::fabs(x - pt.x) > DOUBLE_EPSILON ||
                std::fabs(y - pt.y) > DOUBLE_EPSILON);
    }

    inline bool operator==(const Point2& pt) const {
        return (std::fabs(x - pt.x) < DOUBLE_EPSILON &&
                std::fabs(y - pt.y) < DOUBLE_EPSILON);
    }

    friend std::ostream& operator<<(std::ostream &strm, const Point2 &pt) {
        strm << "(" << pt.x << "," << pt.y << ") ";
        return strm;
    }
};

Point2 Arithmetic_Export operator*(double scale, const Point2& v);

Vector2 Arithmetic_Export operator-(const Point2& pt1, const Point2& pt2);

Point2 Arithmetic_Export operator+(const Point2& pt, const Vector2& v);

Point2 Arithmetic_Export operator-(const Point2& pt, const Vector2& v);

MED_IMG_END_NAMESPACE

#endif
