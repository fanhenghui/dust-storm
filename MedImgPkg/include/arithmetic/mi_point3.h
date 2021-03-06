#ifndef MEDIMGARITHMETIC_MI_POINT3_H
#define MEDIMGARITHMETIC_MI_POINT3_H

#include <ostream>
#include "arithmetic/mi_vector3.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Point3 {
public:
    double x, y, z;
    static const Point3 S_ZERO_POINT;

public:
    ~Point3() {}

    Point3() : x(0), y(0), z(0) {}

    Point3(double x1, double y1, double z1) : x(x1), y(y1), z(z1) {}

    Point3(const Point3& pt) {
        x = pt.x;
        y = pt.y;
        z = pt.z;
    }

    inline Point3& operator+=(const Vector3& v) {
        this->x = this->x + v.x;
        this->y = this->y + v.y;
        this->z = this->z + v.z;
        return *this;
    }

    inline Point3& operator-=(const Vector3& v) {
        this->x = this->x - v.x;
        this->y = this->y - v.y;
        this->z = this->z - v.z;
        return *this;
    }

    inline Point3& operator*=(double scale) {
        x *= scale;
        y *= scale;
        z *= scale;
        return *this;
    }

    inline Point3 operator*(double scale) const {
        return Point3(x * scale, y * scale, z * scale);
    }

    inline Point3& operator=(const Point3& pt) {
        this->x = pt.x;
        this->y = pt.y;
        this->z = pt.z;
        return *this;
    }

    inline bool operator!=(const Point3& pt) const {
        return (std::fabs(x - pt.x) > DOUBLE_EPSILON ||
                std::fabs(y - pt.y) > DOUBLE_EPSILON ||
                std::fabs(z - pt.z) > DOUBLE_EPSILON);
    }

    inline bool operator==(const Point3& pt) const {
        return (std::fabs(x - pt.x) < DOUBLE_EPSILON &&
                std::fabs(y - pt.y) < DOUBLE_EPSILON &&
                std::fabs(z - pt.z) < DOUBLE_EPSILON);
    }

    friend std::ostream& operator<<(std::ostream &strm, const Point3& pt) {
        strm << "(" << pt.x << "," << pt.y << "," << pt.z << ") ";
        return strm;
    }
};

Point3 Arithmetic_Export operator*(double scale, const Point3& v);

Vector3 Arithmetic_Export operator-(const Point3& pt1, const Point3& pt2);

Point3 Arithmetic_Export operator+(const Point3& pt, const Vector3& v);

Point3 Arithmetic_Export operator-(const Point3& pt, const Vector3& v);

MED_IMG_END_NAMESPACE

#endif
