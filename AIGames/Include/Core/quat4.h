#ifndef ARTHMETIC_QUATERNION_4D_H_
#define ARTHMETIC_QUATERNION_4D_H_

#include "Core/vector3.h"
#include "Core/matrix4.h"

class  Quat4
{
public:
	double x, y, z, w;
	bool m_bNorm;

public:
	~Quat4()
	{
	}

	Quat4()
		: w(0), x(1), y(0), z(0)
	{
	}

	Quat4(double angle, const Vector3& axis)
		: w(std::cos(angle / 2)) , m_bNorm(true)
	{
		double mag = axis.magnitude();
		if (mag < DOUBLE_EPSILON)
		{
			x = 0;
			y = 0;
			z = 0;
		}
		else
		{
			double sinV = std::sin(angle / 2);
			x = sinV * axis.x / mag;
			y = sinV * axis.y / mag;
			z = sinV * axis.z / mag;
		}
	}


	Quat4(const Quat4& quat)
		: w(quat.w),x(quat.x),y(quat.y),z(quat.z),m_bNorm(quat.m_bNorm)
	{

	}

	inline Quat4 operator-() const
	{
		Quat4 quat;
		quat.w = -w;
		quat.x = -x;
		quat.y = -y;
		quat.z = -z;
		return quat;
	}

	inline Quat4 operator*(const Quat4& quat) const
	{
		Quat4 quatOut;
		quatOut.w = w * quat.w - x * quat.x - y * quat.y - z * quat.z;
		quatOut.x = w * quat.x + x * quat.w + y * quat.z - z * quat.y;
		quatOut.y = w * quat.y - x * quat.z + y * quat.w + z * quat.x;
		quatOut.z = w * quat.z + x * quat.y - y * quat.x + z * quat.w;

		return quatOut;
	}

	inline Quat4& operator*=(const Quat4& quat)
	{
		*this = *this * quat;
		return *this;
	}

	Matrix4 to_matrix() const
	{
		Quat4 quat(*this);
		quat.normalize();

		double ww = quat.w * quat.w;
		double xx = quat.x * quat.x;
		double yy = quat.y * quat.y;
		double zz = quat.z * quat.z;
		double xy = 2 * quat.x * quat.y;
		double yz = 2 * quat.y * quat.z;
		double xz = 2 * quat.x * quat.z;
		double wx = 2 * quat.w * quat.x;
		double wy = 2 * quat.w * quat.y;
		double wz = 2 * quat.w * quat.z;

		return Matrix4(
			ww + xx - yy - zz, xy + wz, xz - wy, 0,
			xy - wz, ww - xx + yy - zz, yz + wx, 0,
			xz + wy, yz - wx, ww - xx - yy + zz, 0,
			0, 0, 0, 1);
	}

	inline bool normalize()
	{
		if (m_bNorm)
		{
			return true;
		}
		else
		{
			double norm = std::sqrt(w * w + x * x + y * y + z * z);
			if (DOUBLE_EPSILON > norm)
			{
				return false;
			}
			else
			{
				double norminv = 1 / norm;
				w *= norminv;
				x *= norminv;
				y *= norminv;
				z *= norminv;
				return true;
			}
		}
	}

	inline bool operator==(const Quat4& quat) const
	{
		return (std::abs(w - quat.w) < DOUBLE_EPSILON &&
			std::abs(x - quat.x) < DOUBLE_EPSILON &&
			std::abs(y - quat.y) < DOUBLE_EPSILON &&
			std::abs(z - quat.z) < DOUBLE_EPSILON);
	}

	inline bool operator!=(const Quat4& quat) const
	{
		return (std::abs(w - quat.w) > DOUBLE_EPSILON ||
			std::abs(x - quat.x) > DOUBLE_EPSILON ||
			std::abs(y - quat.y) > DOUBLE_EPSILON ||
			std::abs(z - quat.z) > DOUBLE_EPSILON);
	}

	inline Quat4& operator=(const Quat4& quat)
	{
		w = quat.w;
		x = quat.x;
		y = quat.y;
		z = quat.z;
		return *this;
	}
};



#endif
