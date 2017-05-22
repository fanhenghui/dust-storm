#pragma once

#include <memory>
#include "Core/point2.h"

class GLTexture2D;
class Mine
{
public:
    Mine();
    ~Mine();

    void reset();

    void draw();

    void set_texture(std::shared_ptr<GLTexture2D> tex);

public:
    Point2 _position;
    std::shared_ptr<GLTexture2D> _tex;
    static double S_WIDTH;
};