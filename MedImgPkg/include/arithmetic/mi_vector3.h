#ifndef MEDIMGARITHMETIC_MI_VECTOR3_H
#define MEDIMGARITHMETIC_MI_VECTOR3_H

#include <ostream>
#include "arithmetic/mi_arithmetic_export.h"
#include "log/mi_logger_util.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Vector3 {
public:
    double x, y, z;
    static const Vector3 S_ZERO_VECTOR;

public:
    ~Vector3() {}

    Vector3() : x(0), y(0), z(0) {}

    Vector3(double x1, double y1, double z1) : x(x1), y(y1), z(z1) {}

    Vector3(const Vector3& v) {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    inline Vector3& operator+=(const Vector3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    inline Vector3 operator+(const Vector3& v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    inline Vector3& operator-=(const Vector3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    inline Vector3 operator-(const Vector3& v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    inline Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    inline Vector3& operator*=(double scale) {
        x *= scale;
        y *= scale;
        z *= scale;
        return *this;
    }

    inline Vector3 operator*(double scale) const {
        return Vector3(x * scale, y * scale, z * scale);
    }

    inline bool operator!=(const Vector3& v) const {
        return (std::fabs(x - v.x) > DOUBLE_EPSILON ||
                std::fabs(y - v.y) > DOUBLE_EPSILON ||
                std::fabs(z - v.z) > DOUBLE_EPSILON);
    }

    inline bool operator==(const Vector3& v) const {
        return (std::fabs(x - v.x) < DOUBLE_EPSILON &&
                std::fabs(y - v.y) < DOUBLE_EPSILON &&
                std::fabs(z - v.z) < DOUBLE_EPSILON);
    }

    inline Vector3& operator=(const Vector3& v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    inline double angle_between(const Vector3& v) const {
        double len = magnitude() * v.magnitude();

        len = (len > DOUBLE_EPSILON) ? len : DOUBLE_EPSILON;

        double dot = dot_product(v) / len;
        dot = (std::min)(dot, 1.0);
        dot = (std::max)(dot, -1.0);
        return std::acos(dot);
    }

    inline Vector3 cross_product(const Vector3& v) const {
        return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    inline double dot_product(const Vector3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    inline double magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    inline void normalize() {
        double len = std::sqrt(x * x + y * y + z * z);

        if (len < DOUBLE_EPSILON) {
            //LoggerUtil::log(MI_WARNING , "Arithmetic" , "vector3's magnitude is 0 ! Get normalize failed.");
            return;
        }

        double leninv = 1.0 / len;
        x *= leninv;
        y *= leninv;
        z *= leninv;
    }

    Vector3 get_normalize() const {
        double len = std::sqrt(x * x + y * y + z * z);

        if (len < DOUBLE_EPSILON) {
            //LoggerUtil::log(MI_WARNING , "Arithmetic" , "vector3's magnitude is 0 ! Get normalize failed.");
            return *this;
        }

        double leninv = 1.0 / len;
        return Vector3(x * leninv, y * leninv, z * leninv);
    }

    inline Vector3 reflect(const Vector3& norm) const {
        return Vector3(*this - norm * (2 * this->dot_product(norm)));
    }

    bool parallel(const Vector3& v) const {
        return this->cross_product(v) == Vector3(0, 0, 0);
    }

    bool orthogonal(const Vector3& v) const {
        return std::fabs(this->dot_product(v)) < DOUBLE_EPSILON;
    }

    friend std::ostream& operator<<(std::ostream &strm, const Vector3 &pt) {
        strm << "(" << pt.x << "," << pt.y << "," << pt.z << ") ";
        return strm;
    }
};

Vector3 Arithmetic_Export operator*(double scale, const Vector3& v);

double Arithmetic_Export angle_between(const Vector3& v1, const Vector3& v2);

Vector3 Arithmetic_Export cross(const Vector3& v1, const Vector3& v2);

double Arithmetic_Export dot_product(const Vector3& v1, const Vector3& v2);

bool Arithmetic_Export parallel(const Vector3& v1, const Vector3& v2);

bool Arithmetic_Export orthogonal(const Vector3& v1, const Vector3& v2);

MED_IMG_END_NAMESPACE

#endif