#include "gl/glew.h"
#include "gl/glut.h"
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

extern int load_dicom_series(std::vector<std::string>& files ,
    std::shared_ptr<ImageDataHeader>& header,
    std::shared_ptr<ImageData>& img);

extern int get_nodule_set(const std::string& xml_file, std::vector<std::vector<Nodule>>& nodules , std::string& series_uid);

extern int resample_z(std::vector<std::vector<Nodule>>& nodules , std::shared_ptr<ImageDataHeader>& header , bool save_slice_location_less);

extern void cal_nodule_aabb(std::vector <std::vector<Nodule>>& nodules );

extern void get_same_region_nodules(std::vector <std::vector<Nodule>>::iterator reader , Nodule& target , std::vector <std::vector<Nodule>>& nodules, std::vector<Nodule*>& same_nodules, std::vector<int>& reader_id ,  float region_percent);

typedef ScanLineAnalysis<unsigned char>::Pt2 PT2;
extern void scan_contour_to_mask(std::vector<Point3>& pts , std::shared_ptr<ImageData> mask, unsigned char label);

extern int contour_to_mask(std::vector <std::vector<Nodule>>& nodules , std::shared_ptr<ImageData> mask, float same_nodule_precent , int confidence, int setlogic);

extern int browse_root_xml(const std::string& root , std::vector<std::string>& xml_files );

extern int browse_root_dcm(const std::string& root, std::map<std::string, std::vector<std::string>>& dcm_files );

std::vector<std::vector<Nodule>> _vis_nodules;
int _cur_z = 46;
int _max_z = 133;
int _cur_reader = 0;
int _pre_x = 0;
int _pre_y = 0;

void display()
{
    glViewport(0,0,_width,_height);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT );

    glPushMatrix();

    /*glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0,0,100,0,0,0,0,1,0);*/
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,_width ,0,_height);

    glPointSize(2.0);
    //if (!_pre_contours.empty())
    //{
    //    glColor3f(0.0,1.0,1.0);
    //    glBegin(GL_POINTS);
    //    for (int i = 0 ; i<_pre_contours.size() ; ++i)
    //    {
    //        for (int j = 0 ; j < _pre_contours[i].size() ; ++j)
    //        {
    //            glVertex2d(_pre_contours[i][j].x , _pre_contours[i][j].y);
    //        }
    //    }
    //    glEnd();
    //}

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
                    glBegin(GL_LINES);
                    for (int j = i ; j < pts.size() ; ++j)
                    {
                        if (pts[j].z != _cur_z)
                        {
                            break;
                        }
                        glVertex2d(pts[j].x , pts[j].y);
                    }
                    glVertex2d(pts[i].x , pts[i].y);
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
    default:
        break;
    }

    glutPostRedisplay();
}

void reshape(int x , int y)
{
    if (x == 0 || y == 0)
    {
        return;
    }
    _width = x;
    _height = y;

    glutPostRedisplay();
}

void motion(int x , int y)
{
    int delta = y - _pre_y;
    _cur_z += delta;
    if (_cur_z > _max_z)
    {
        _cur_z = _max_z;
    }
    if (_cur_z < 0)
    {
        _cur_z = 0;
    }

    _pre_x = x;
    _pre_y = y;

    std::cout << "Current Z : "<< _cur_z << std::endl;
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

int ExtractMaskVis(int argc , char* argv[])
{
    /*arguments list:
    -help : print all argument
    -data <path> : dicom data root(.dcm)
    -annotation <path> : annotation root(.xml)
    -output <path] : save mask root
    -compress : if mask is compressed 
    -slicelocation <less/greater> : default is less

    -crosspercent<0.1~1> default 0.7
    -confidence<1~4> default is 2
    -setlogic<inter/union> default is inter 
    */


    std::string dcm_direction;
    std::string annotation_direction;
    std::string output_direction;
    bool compressed = false;
    bool save_slice_location_less = true;
    float cross_nodule_percent = 0.7f;
    int confidence = 2;
    int setlogic = 0;//0 for intersection 1 for union

    if (argc == 1)
    {
        LOG_OUT("invalid arguments!\n");
        LOG_OUT("targuments list:\n");
        LOG_OUT("\t-data <path> : DICOM data root(.dcm)\n");
        LOG_OUT("\t-annotation <path> : annotation root(.xml)\n");
        LOG_OUT("\t-output <path] : save mask root\n");
        LOG_OUT("\t-compress : if mask is compressed\n");
        LOG_OUT("\t-slicelocation <less/greater> : default is less\n");
        LOG_OUT("\t-crosspercent<0.1~1> default 0.7\n");
        LOG_OUT("\t-confidence<1~4> default is 2\n");
        LOG_OUT("\t-setlogic<inter/union> default is inter\n");
        return -1;
    }
    else
    {
        for (int i = 1; i< argc ; ++i)
        {
            if (std::string(argv[i]) == "-help")
            {
                LOG_OUT("arguments list:\n");
                LOG_OUT("\t-data <path> : DICOM data root(.dcm)\n");
                LOG_OUT("\t-annotation <path> : annotation root(.xml)\n");
                LOG_OUT("\t-output <path] : save mask root\n");
                LOG_OUT("\t-compress : if mask is compressed\n");
                LOG_OUT("\t-slicelocation <less/greater> : default is less\n");
                LOG_OUT("\t-crosspercent<0.1~1> default 0.7\n");
                LOG_OUT("\t-confidence<1~4> default is 2\n");
                LOG_OUT("\t-setlogic<inter/union> default is inter\n");
                return 0;
            }
            if (std::string(argv[i]) == "-data")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }
                dcm_direction = std::string(argv[i+1]);

                ++i;
            }
            else if (std::string(argv[i]) == "-annotation")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }

                annotation_direction = std::string(argv[i+1]);
                ++i;
            }
            else if (std::string(argv[i])== "-output")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }

                output_direction = std::string(argv[i+1]);
                ++i;
            }
            else if (std::string(argv[i]) == "-compress")
            {
                compressed = true;
            }
            else if (std::string(argv[i]) == "-slicelocation")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }

                if(std::string(argv[i+1]) == "less")
                {
                    save_slice_location_less = true;
                }
                else if(std::string(argv[i+1]) == "greater")
                {
                    save_slice_location_less =false;
                }
                else
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }
                ++i;
            }
            else if (std::string(argv[i]) == "-crosspercent")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }
                StrNumConverter<float> conv;
                cross_nodule_percent = conv.to_num(std::string(argv[i+1]));
                ++i;
            }
            else if (std::string(argv[i]) == "-confidence")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }
                StrNumConverter<int> conv;
                confidence = conv.to_num(std::string(argv[i+1]));
                ++i;
            }
            else if (std::string(argv[i]) == "-setlogic")
            {
                if (i+1 > argc-1)
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }

                if(std::string(argv[i+1]) == "inter")
                {
                    setlogic = 0;
                }
                else if(std::string(argv[i+1]) == "union")
                {
                    setlogic = 1;
                }
                else
                {
                    LOG_OUT(  "invalid arguments!\n");
                    return -1;
                }
                ++i;
            }
        }
    }

    if (dcm_direction.empty() || annotation_direction.empty() || output_direction.empty())
    {
        LOG_OUT(  "invalid empty direction!\n");
        return -1;
    }

    std::map<std::string , std::vector<std::string>> dcm_files;
    if (0 != browse_root_dcm(dcm_direction , dcm_files))
    {
        LOG_OUT(  "browse dicom direction failed!\n");
        return -1;
    }

    std::vector<std::string> xml_files;
    if (0 != browse_root_xml(annotation_direction , xml_files))
    {
        LOG_OUT(  "browse annotation direction failed!\n");
        return -1;
    }

    for (auto it = xml_files.begin() ; it != xml_files.end() ; ++it)
    {
        LOG_OUT(  "parse annotation file : " + *it + " >>>\n");

        //parse nodule annotation file
        std::vector<std::vector<Nodule>> nodules;
        std::string series_uid;
        if(0 !=  get_nodule_set(*it , nodules , series_uid))
        {
            LOG_OUT(  "parse nodule annotation failed!\n");
            return -1;
        }

        LOG_OUT("series UID : " + series_uid + "\n");

        auto it_dcm = dcm_files.find(series_uid);
        if (it_dcm == dcm_files.end())
        {
            LOG_OUT(  "can't find dcm files!\n");
            continue;
        }

        LOG_OUT( "loading DICOM files to get image information :  >>>\n");

        //slice location to pixal coordinate
        std::shared_ptr<ImageDataHeader> data_header;
        std::shared_ptr<ImageData> volume_data;
        if(0 != load_dicom_series(it_dcm->second ,data_header,volume_data ))
        {
            LOG_OUT(  "load series failed!\n");
            return -1;
        }

        //resample position z from slice location to pixel coordinate
        if( 0 != resample_z(nodules , data_header , save_slice_location_less) )
        {
            LOG_OUT( "resample z failed!\n");
            return -1;
        }

        LOG_OUT( "convert contour to mask >>>\n");

        //calculate aabb for extract mask
        cal_nodule_aabb(nodules);

        //contour to mask
        std::shared_ptr<ImageData> mask(new ImageData);
        volume_data->shallow_copy(mask.get());
        if (0 != contour_to_mask(nodules , mask , cross_nodule_percent ,confidence, setlogic))
        {
            LOG_OUT( "convert contour to mask failed!\n");
            return -1;
        }

        //save mask
        std::string output;
        if (compressed)
        {
            output = output_direction+ series_uid + ".rle";
        }
        else
        {
            output = output_direction+ series_uid + ".raw";
        }
        if(0 != save_mask( mask , output ,compressed))
        {
            LOG_OUT( "save mask failed!\n");
            return -1;
        }

        LOG_OUT( "extract mask done.\n");

        _vis_nodules = nodules;
    }

    LOG_OUT(  "done.\n");


    glutInit(&argc , argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(_width,_height);

    glutCreateWindow("Test Scan Line");

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

    glutMainLoop(); 

    return 0;
};