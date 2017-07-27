#include "gl/glew.h"
#include "gl/freeglut.h"
#include "gl/GLU.h"

#include <string>
#include <fstream>
#include <vector>

#include "MedImgCommon/mi_common_file_util.h"
#include "MedImgArithmetic/mi_scan_line_analysis.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_aabb.h"
#include "MedImgArithmetic/mi_intersection_test.h"
#include "MedImgCommon/mi_string_number_converter.h"
#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_run_length_operator.h"
#include "Ext/pugixml/pugixml.hpp"
#include "Ext/pugixml/pugiconfig.hpp"

#include "mi_extract_mask_common.h"

using namespace medical_imaging;

int _width = 512;
int _height = 512;

//////////////////////////////////////////////////////////////////////////
#define LOG_OUT(info) std::cout << info; 
//////////////////////////////////////////////////////////////////////////


extern int save_mask(std::shared_ptr<ImageData> mask , const std::string& path , bool compressed);

extern int load_dicom_series(std::vector<std::string>& files , std::shared_ptr<ImageDataHeader>& header, std::shared_ptr<ImageData>& img);

extern int get_nodule_set(const std::string& xml_file, std::vector<std::vector<Nodule>>& nodules , std::string& series_uid);

extern int resample_z(std::vector<std::vector<Nodule>>& nodules , std::shared_ptr<ImageDataHeader>& header , bool save_slice_location_less);

extern void cal_nodule_aabb(std::vector <std::vector<Nodule>>& nodules );

extern void get_same_region_nodules(std::vector <std::vector<Nodule>>::iterator reader , Nodule& target , std::vector <std::vector<Nodule>>& nodules, std::vector<Nodule*>& same_nodules, std::vector<int>& reader_id ,  float region_percent);

typedef ScanLineAnalysis<unsigned char>::Pt2 PT2;
extern void scan_contour_to_mask(std::vector<Point3>& pts , std::shared_ptr<ImageData> mask, unsigned char label);

extern int contour_to_mask(std::vector <std::vector<Nodule>>& nodules , std::shared_ptr<ImageData> mask, float same_nodule_precent , int confidence, int pixel_confidence , int setlogic);

extern int browse_root_xml(const std::string& root , std::vector<std::string>& xml_files );

extern int browse_root_dcm(const std::string& root, std::map<std::string, std::vector<std::string>>& dcm_files );

extern int extract_mask(int argc , char* argv[] , std::shared_ptr<ImageData>& last_img , std::shared_ptr<ImageDataHeader>& last_header , std::vector<std::vector<Nodule>>& last_nodule);

std::vector<std::vector<Nodule>> _vis_nodules;
std::shared_ptr<ImageData> _img_data;
std::shared_ptr<ImageDataHeader> _data_header;
std::shared_ptr<unsigned char> _buffer;
int _cur_z = 46;
int _max_z = 133;
int _cur_reader = 0;
int _pre_x = 0;
int _pre_y = 0;

const float PRESET_CT_LUNGS_WW = 1500;
const float PRESET_CT_LUNGS_WL = -400;

float _ww = PRESET_CT_LUNGS_WW;
float _wl = PRESET_CT_LUNGS_WL;

void calculate_raw_image_buffer();

void display()
{
    glViewport(0,0,_width,_height);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT );

    calculate_raw_image_buffer();
    glDrawPixels(_width , _height , GL_RGB , GL_UNSIGNED_BYTE , _buffer.get());

    glPushMatrix();

    /*glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0,0,100,0,0,0,0,1,0);*/
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,_width ,0,_height);

    glPointSize(2.0);
    if (!_vis_nodules.empty())
    {
        const Vector3 colors[4] = {
            Vector3(1.0,0.0,0.0),
            Vector3(1.0,1.0,0.0),
            Vector3(1.0,0.0,1.0),
            Vector3(0.0,0.0,1.0)
        };

        for (int reader = 0; reader <_vis_nodules.size() ; ++reader)
        {
            if (_cur_reader !=4 && reader != _cur_reader)
            {
                continue;
            }

            for(auto it = (_vis_nodules[reader]).begin()  ; it != (_vis_nodules[reader]).end() ; ++it)
            {
                Nodule& nodule = *it;
                std::vector<Point3>& pts = nodule._points;
                int i = 0;
                for ( ; i<pts.size() ; ++i)
                {
                    if (pts[i].z == _cur_z)
                    {
                        break;
                    }
                }

                if (i<pts.size())
                {
                    glColor3d(colors[reader].x , colors[reader].y , colors[reader].z);
                    glBegin(GL_LINE_STRIP);
                    for (int j = i ; j < pts.size() ; ++j)
                    {
                        if (pts[j].z != _cur_z)
                        {
                            break;
                        }
                        glVertex2d(pts[j].x , _height - pts[j].y - 1);
                    }
                    glVertex2d(pts[i].x , _height -pts[i].y - 1);
                    glEnd();
                }
            }
        }

    }

    glPopMatrix();

    glutSwapBuffers();
}

void keybord(unsigned char key , int x, int y)
{
    switch(key)
    {
    case '0':
        {
            _cur_reader= 0;
            break;
        }
    case '1':
        {
            _cur_reader= 1;
            break;
        }
    case '2':
        {
            _cur_reader= 2;
            break;
        }
    case '3':
        {
            _cur_reader= 3;
            break;
        }
    case '4':
        {
            _cur_reader= 4;
            break;
        }
    case 'h':
        {
            std::cout << "\n************  user manual ************\n";
            std::cout << "\tnum 0 : show reader 1's annotation.\n";
            std::cout << "\tnum 1 : show reader 2's annotation.\n";
            std::cout << "\tnum 2 : show reader 3's annotation.\n";
            std::cout << "\tnum 3 : show reader 4's annotation.\n";
            std::cout << "\tnum 4 : show all annotation.\n";
            std::cout << "\tmouse wheel for paging.\n";
            std::cout << "\tmouse click&motion for windowing.\n";
            std::cout << "************  user manual ************\n";
            break;
        }
    default:
        break;
    }

    glutPostRedisplay();
}

void reshape(int x , int y)
{
    glutPostRedisplay();
}

void motion(int x , int y)
{
    int delta_y = y - _pre_y;
    int delta_x = x - _pre_x;

    _wl += delta_y ;
    _ww += delta_x ;
    if (_ww < 1)
    {
        _ww = 1;
    }

    _pre_x = x;
    _pre_y = y;

    glutPostRedisplay();

}

void mouse(int btn , int status , int x , int y)
{
    if (status == GLUT_DOWN)
    {
        
    }
    else if(status == GLUT_UP)
    {

    }

    _pre_x = x;
    _pre_y = y;
    glutPostRedisplay();
}

void mousewheel(int btn , int dir, int x , int y)
{
    _cur_z += dir;
    if (_cur_z > _max_z-1)
    {
        _cur_z = _max_z-1;
    }
    if (_cur_z < 0)
    {
        _cur_z = 0;
    }
    std::cout << "Current Z : "<< _cur_z << std::endl;

    glutPostRedisplay();
}


void calculate_raw_image_buffer()
{
    const float min_wl = _wl - _img_data->_intercept - _ww*0.5f;

    if (_img_data->_data_type == DataType::USHORT)
    {
        unsigned short* volume_data = (unsigned short*)_img_data->get_pixel_pointer();
        for (int y  = 0 ; y <_height; ++y)
        {
            int yy = _height - y - 1;
            for (int x = 0; x < _width ; ++x)
            {
                unsigned short v = volume_data[_cur_z*_width*_height + yy*_width + x];
                float v0 = ((float)v  - min_wl)/_ww;
                v0 = v0 > 1.0f ? 1.0f : v0;
                v0 = v0 < 0.0f ? 0.0f : v0;
                unsigned char rgb = static_cast<unsigned char>(v0*255.0f);
                _buffer.get()[(y*_width + x)*3] = rgb;
                _buffer.get()[(y*_width + x)*3+1] = rgb;
                _buffer.get()[(y*_width + x)*3+2] = rgb;
            }
        }
    }
    else if (_img_data->_data_type == DataType::SHORT)
    {
        short* volume_data = (short*)_img_data->get_pixel_pointer();
        for (int y  = 0 ; y <_height ; ++y)
        {
            int yy = _height - y - 1;
            for (int x = 0; x < _width ; ++x)
            {
                short v = volume_data[_cur_z*_width*_height + yy*_width + x];
                float v0 = ((float)v   - min_wl)/_ww;
                v0 = v0 > 1.0f ? 1.0f : v0;
                v0 = v0 < 0.0f ? 0.0f : v0;
                unsigned char rgb = static_cast<unsigned char>(v0*255.0f);
                _buffer.get()[(y*_width + x)*3] = rgb;
                _buffer.get()[(y*_width + x)*3+1] = rgb;
                _buffer.get()[(y*_width + x)*3+2] = rgb;
            }
        }
    }
}

int logic_vis(int argc , char* argv[])
{
    if(0 != extract_mask(argc , argv , _img_data , _data_header , _vis_nodules))
    {
        return -1;
    }

    _width =  _img_data->_dim[0];
    _height =  _img_data->_dim[1];
    _max_z = _img_data->_dim[2];
    _cur_z = _img_data->_dim[2]/2;
    _buffer.reset(new  unsigned char[_width*_height*3] , std::default_delete<unsigned char[]>());

    glutInit(&argc , argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(_width,_height);

    glutCreateWindow(_data_header->series_uid.c_str());

    if ( GLEW_OK != glewInit())
    {
        std::cout <<"Init glew failed!\n";
        return -1;
    }

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keybord);
    glutMouseWheelFunc(mousewheel);

    glutMainLoop(); 

    return 0;
};