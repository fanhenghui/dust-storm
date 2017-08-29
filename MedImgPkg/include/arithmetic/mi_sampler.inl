
template <class T>
float Sampler<T>::sample_1d_nearst(float idx, unsigned int len, T* data) const {
    const unsigned int uIdx = (unsigned int)idx;
    return data[uIdx];
}

template <class T>
float Sampler<T>::sample_1d_linear(float idx, unsigned int len, T* data) const {
    if (idx >= (float)(len - 1) - FLOAT_EPSILON) {
        return (float)data[len - 1];
    } else {
        const unsigned int uIdxLower = (unsigned int)idx;
        const unsigned int uIdxUpper = uIdxLower + 1;
        float fLower = idx - (float)uIdxLower;
        float fUpper = 1.0f - fLower;
        return data[uIdxLower] * fLower + data[uIdxUpper] * fUpper;
    }
}

template <class T>
float Sampler<T>::sample_2d_nearst(float x, float y, unsigned int uiWidth,
                                   unsigned int uiHeight, T* data) const {
    const unsigned int uX = (unsigned int)x;
    const unsigned int uY = (unsigned int)y;
    return (float)(data[uY * uiWidth + uX]);
}

template <class T>
float Sampler<T>::sample_2d_linear(float x, float y, unsigned int uiWidth,
                                   unsigned int uiHeight, T* data) const {
    unsigned int uiXAdjust = 1;

    if (x >= (float)(uiWidth - 1) - FLOAT_EPSILON) {
        x = (float)(uiWidth - 1);
        uiXAdjust = 0;
    }

    unsigned int uiYAdjust = uiWidth;

    if (y >=
            (float)(uiHeight - 1) -
            FLOAT_EPSILON) { // TODO ��������==�� ����������y����Ϊ������ʱ������ȥ��֧
        y = (float)(uiHeight - 1);
        uiYAdjust = 0;
    }

    const unsigned int uX = (unsigned int)x;
    const unsigned int uY = (unsigned int)y;
    const float fX1 = x - (float)uX;
    const float fX0 = 1.0f - fX1;
    const float fY1 = y - (float)uY;
    const float fY0 = 1.0f - fY1;

    const unsigned int uX0Offset = uY * uiWidth + uX;
    const unsigned int uX1Offset = uX0Offset + uiYAdjust;

    const float fX00_01 =
        (float)data[uX0Offset] * fX0 + (float)data[uX0Offset + uiXAdjust] * fX1;
    const float fX10_11 =
        (float)data[uX1Offset] * fX0 + (float)data[uX1Offset + uiXAdjust] * fX1;

    const float fY = fX00_01 * fY0 + fX10_11 * fY1;

    return fY;
}

template <class T>
float Sampler<T>::sample_3d_nearst(float x, float y, float z,
                                   unsigned int uiWidth, unsigned int uiHeight,
                                   unsigned int uiDepth, T* data) const {
    const unsigned int uX = (unsigned int)x;
    const unsigned int uY = (unsigned int)y;
    const unsigned int uZ = (unsigned int)z;
    T v = data[uZ * uiWidth * uiHeight + uY * uiWidth + uX];
    return (float)v;
}

template <class T>
float Sampler<T>::sample_3d_linear(float x, float y, float z,
                                   unsigned int uiWidth, unsigned int uiHeight,
                                   unsigned int uiDepth, T* data) const {
    unsigned int uiXAdjust = 1;

    if (x >= (float)(uiWidth - 1) - FLOAT_EPSILON) {
        x = (float)(uiWidth - 1);
        uiXAdjust = 0;
    }

    unsigned int uiYAdjust = uiWidth;

    if (y >=
            (float)(uiHeight - 1) -
            FLOAT_EPSILON) { // TODO ��������==�� ����������y����Ϊ������ʱ������ȥ��֧
        y = (float)(uiHeight - 1);
        uiYAdjust = 0;
    }

    unsigned int uiZAdjust = uiWidth * uiHeight;

    if (z >=
            (float)(uiDepth - 1) -
            FLOAT_EPSILON) { // TODO ��������==�� ����������z����Ϊ������ʱ������ȥ��֧
        z = (float)(uiDepth - 1);
        uiZAdjust = 0;
    }

    const unsigned int uX = (unsigned int)x;
    const unsigned int uY = (unsigned int)y;
    const unsigned int uZ = (unsigned int)z;
    const float fX1 = x - (float)uX;
    const float fX0 = 1.0f - fX1;
    const float fY1 = y - (float)uY;
    const float fY0 = 1.0f - fY1;
    const float fZ1 = z - (float)uZ;
    const float fZ0 = 1.0f - fZ1;

    const unsigned int uZ0Offset = uZ * uiWidth * uiHeight + uY * uiWidth + uX;
    const unsigned int uZ1Offset = uZ0Offset + uiZAdjust;

    // X direction
    const float fX000_100 =
        (float)data[uZ0Offset] * fX0 + (float)data[uZ0Offset + uiXAdjust] * fX1;
    const float fX010_110 = (float)data[uZ0Offset + uiYAdjust] * fX0 +
                            (float)data[uZ0Offset + uiYAdjust + uiXAdjust] * fX1;
    const float fX001_101 =
        (float)data[uZ1Offset] * fX0 + (float)data[uZ1Offset + uiXAdjust] * fX1;
    const float fX011_111 = (float)data[uZ1Offset + uiYAdjust] * fX0 +
                            (float)data[uZ1Offset + uiYAdjust + uiXAdjust] * fX1;

    // Y direction
    const float fY00 = fX000_100 * fY0 + fX010_110 * fY1;
    const float fY11 = fX001_101 * fY0 + fX011_111 * fY1;

    // Z direction
    const float fZ = fY00 * fZ0 + fY11 * fZ1;

    return fZ;
}