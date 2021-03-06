#include "mi_arithmetic_utils.h"

MED_IMG_BEGIN_NAMESPACE

Matrix4f ArithmeticUtils::convert_matrix(const Matrix4& mat) {
    Vector4f col0((float)mat.m[0][0], (float)mat.m[0][1], (float)mat.m[0][2],
                  (float)mat.m[0][3]);
    Vector4f col1((float)mat.m[1][0], (float)mat.m[1][1], (float)mat.m[1][2],
                  (float)mat.m[1][3]);
    Vector4f col2((float)mat.m[2][0], (float)mat.m[2][1], (float)mat.m[2][2],
                  (float)mat.m[2][3]);
    Vector4f col3((float)mat.m[3][0], (float)mat.m[3][1], (float)mat.m[3][2],
                  (float)mat.m[3][3]);

    return Matrix4f(col0, col1, col2, col3);
}

Vector3f ArithmeticUtils::convert_vector(const Vector3& v) {
    return Vector3f((float)v.x, (float)v.y, (float)v.z);
}

Vector3f ArithmeticUtils::convert_point(const Point3& v) {
    return Vector3f((float)v.x, (float)v.y, (float)v.z);
}

Point2 ArithmeticUtils::dc_to_ndc(const Point2& pt_dc, int width, int height) {
    double x = (pt_dc.x + 0.5) / (double)width;
    double y = (pt_dc.y + 0.5) / (double)height;
    return Point2(x * 2.0 - 1.0, -(y * 2.0 - 1.0));
}

Point2 ArithmeticUtils::ndc_to_dc(Point2 pt_ndc, int width, int height,
                                  int spill_tag) {
    spill_tag = 0;

    if (pt_ndc.x < -1.0) {
        pt_ndc.x = -1.0;
        spill_tag = 1;
    } else if (pt_ndc.x > 1.0) {
        pt_ndc.x = 1.0;
        spill_tag = 1;
    }

    if (pt_ndc.y < -1.0) {
        pt_ndc.y = -1.0;
        spill_tag = 2;
    } else if (pt_ndc.y > 1.0) {
        pt_ndc.y = 1.0;
        spill_tag = 2;
    }

    double x = (pt_ndc.x + 1.0) * 0.5;
    double y = (-pt_ndc.y + 1.0) * 0.5;
    x = x * (double)width - 0.5;
    y = y * (double)height - 0.5;

    return Point2((int)x, (int)y);
}

Point2 ArithmeticUtils::ndc_to_dc(Point2 pt_ndc, int width, int height) {
    double x = (pt_ndc.x + 1.0) * 0.5;
    double y = (-pt_ndc.y + 1.0) * 0.5;
    x = x * (double)width - 0.5;
    y = y * (double)height - 0.5;

    return Point2((int)x, (int)y);
}

Point2 ArithmeticUtils::ndc_to_dc_decimal(Point2 pt_ndc, int width,
        int height) {
    double x = (pt_ndc.x + 1.0) * 0.5;
    double y = (-pt_ndc.y + 1.0) * 0.5;
    x = x * (double)width;
    y = y * (double)height;

    return Point2(x, y);
}

bool ArithmeticUtils::check_in_bound(const Point3& pt, const Point3& bound) {
    if (pt.x < -FLOAT_EPSILON || pt.x > bound.x + FLOAT_EPSILON || 
        pt.y < -FLOAT_EPSILON || pt.y > bound.y + FLOAT_EPSILON|| 
        pt.z < -FLOAT_EPSILON || pt.z > bound.z + FLOAT_EPSILON) {
        return false;
    } else {
        return true;
    }
}

int ArithmeticUtils::get_valid_region(const unsigned int (&dim)[3],
                                       const Sphere& sphere,
                                       unsigned int (&ubegin)[3],
                                       unsigned int (&uend)[3]) {
    const Point3 min =
        sphere._center - Vector3(sphere._radius, sphere._radius, sphere._radius);
    const Point3 max =
        sphere._center + Vector3(sphere._radius, sphere._radius, sphere._radius);

    int begin[3] = {0,0,0};
    int end[3] = {0,0,0};
    int tmp = 0;
    
    tmp = static_cast<int>(min.x + 0.5);
    begin[0] = static_cast<int>(tmp);
    tmp = static_cast<int>(min.y + 0.5);
    begin[1] = static_cast<int>(tmp);
    tmp = static_cast<int>(min.z + 0.5);
    begin[2] = static_cast<int>(tmp);

    tmp = static_cast<int>(max.x + 0.5);
    end[0] = static_cast<int>(tmp);
    tmp = static_cast<int>(max.y + 0.5);
    end[1] = static_cast<int>(tmp);
    tmp = static_cast<int>(max.z + 0.5);
    end[2] = static_cast<int>(tmp);

    int bound_status = 0;
    if (begin[0] >= (int)dim[0] - 1 || begin[1] >= (int)dim[1] - 1 || begin[2] >= (int)dim[2] - 1 ) {
        bound_status = -1;
    }
    if (end[0] <= 0 || end[1] <= 0 || end[2] <= 0) {
        bound_status = -1;
    }

    for (int i = 0; i < 3; ++i) {
        if (begin[i] > (int)dim[i] - 1) {
            begin[i] = (int)dim[i] - 1;
        }
        if (begin[i] < 0) {
            begin[i] = 0;
        }

        if (end[i] > (int)dim[i] - 1) {
            end[i] = (int)dim[i] - 1;
        }
        if (end[i] < 0) {
            end[i] = 0;
        }
    }

    for (int i = 0; i < 3; ++i) {
        ubegin[i] = (unsigned int)begin[i];
        uend[i] = (unsigned int)end[i];
    }

    return bound_status;
}

int ArithmeticUtils::get_valid_region(const unsigned int (&dim)[3],
                                       const Ellipsoid& ellipsoid,
                                       unsigned int (&ubegin)[3],
                                       unsigned int (&uend)[3]) {
    const double radius =
        std::max(std::max(ellipsoid._a, ellipsoid._b), ellipsoid._c);
    const Point3 min = ellipsoid._center - Vector3(radius, radius, radius);
    const Point3 max = ellipsoid._center + Vector3(radius, radius, radius);

    int begin[3] = {0,0,0};
    int end[3] = {0,0,0};
    int tmp = 0;

    tmp = static_cast<int>(min.x + 0.5);
    begin[0] = static_cast<int>(tmp);
    tmp = static_cast<int>(min.y + 0.5);
    begin[1] = static_cast<int>(tmp);
    tmp = static_cast<int>(min.z + 0.5);
    begin[2] = static_cast<int>(tmp);

    tmp = static_cast<int>(max.x + 0.5);
    end[0] = static_cast<int>(tmp);
    tmp = static_cast<int>(max.y + 0.5);
    end[1] = static_cast<int>(tmp);
    tmp = static_cast<int>(max.z + 0.5);
    end[2] = static_cast<int>(tmp);

    int bound_status = 0;
    if (begin[0] >= (int)dim[0] - 1 || begin[1] >= (int)dim[1] - 1 || begin[2] >= (int)dim[2] - 1 ) {
        bound_status = -1;
    }
    if (end[0] <= 0 || end[1] <= 0 || end[2] <= 0) {
        bound_status = -1;
    }

    for (int i = 0; i < 3; ++i) {
        if (begin[i] > (int)dim[i] - 1) {
            begin[i] = (int)dim[i] - 1;
        }
        if (begin[i] < 0) {
            begin[i] = 0;
        }

        if (end[i] > (int)dim[i] - 1) {
            end[i] = (int)dim[i] - 1;
        }
        if (end[i] < 0) {
            end[i] = 0;
        }
    }

    for (int i = 0; i < 3; ++i) {
        ubegin[i] = (unsigned int)begin[i];
        uend[i] = (unsigned int)end[i];
    }

    return bound_status;
}

MED_IMG_END_NAMESPACE