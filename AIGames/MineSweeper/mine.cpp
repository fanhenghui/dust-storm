#include "mine.h"

#include "Ext/gl/glew.h"
#include "Core/gl_texture_2d.h"
#include "Core/gl_utils.h"

#define WIDTH 0.1

Mine::Mine()
{

}

Mine::~Mine()
{

}

void Mine::set_position(const Point2& pt)
{
    _position = pt;
}

void Mine::draw()
{
    CHECK_GL_ERROR;

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    
    CHECK_GL_ERROR;
    _tex->bind();
    glEnable(GL_TEXTURE_2D);
    CHECK_GL_ERROR;
    glBegin(GL_QUADS);

    glTexCoord2d(0, 0);
    glVertex2d(_position.x - WIDTH*0.5, _position.y - WIDTH*0.5);

    glTexCoord2d(0, 1);
    glVertex2d(_position.x - WIDTH*0.5, _position.y + WIDTH*0.5);

    glTexCoord2d(1, 1);
    glVertex2d(_position.x + WIDTH*0.5, _position.y + WIDTH*0.5);

    glTexCoord2d(1, 0);
    glVertex2d(_position.x + WIDTH*0.5, _position.y - WIDTH*0.5);

    glEnd();

    //glDisable(GL_TEXTURE_2D);

    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;
}

void Mine::set_texture(std::shared_ptr<GLTexture2D> tex)
{
    _tex = tex;
}
