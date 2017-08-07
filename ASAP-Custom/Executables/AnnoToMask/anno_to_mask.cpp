#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <vector>

#include "gl/glew.h"
#include "gl/freeglut.h"
#include "gl/GLU.h"


#include "boost/filesystem.hpp"

#include "Annotation/AnnotationList.h"
#include "Annotation/Annotation.h"
#include "Annotation/AnnotationGroup.h"
#include "Annotation/BinaryRepository.h"

#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "io/multiresolutionimageinterface/MultiResolutionImageFactory.h"
#include "io/openslidefileformat/OpenSlideImageFactory.h"
#include "io/openslidefileformat/OpenSlideImage.h"

#include "scan_line_analysis.h"
#include "arithmetic.h"
#include "connected_domain_analysis.h"
#include "segment_analysis.h"
#include "morphology.h"

//#include "Ext/pugixml/pugiconfig.hpp"
//#include "Ext/pugixml/pugixml.hpp"

int _width = 512;
int _height = 512;
int _img_width = 0;
int _img_height = 0;

unsigned char* _vis_img_buffer = nullptr;
std::vector<std::shared_ptr<Annotation>> _vis_annos;
std::vector<AABB<int>> _vis_aabbs;
std::vector<AABB<int>> _vis_c_aabbs;
std::vector<AABB<int>> _vis_e_aabbs;
double _pp_ratio = 1;

static std::ofstream out_log;
class LogSheild
{
public:
    LogSheild(const std::string& log_file, const std::string& start_word)
    {
        out_log.open("anon.log", std::ios::out);
        if (out_log.is_open())
        {
            out_log << start_word;
        }
    }
    ~LogSheild()
    {
        out_log.close();
    }
protected:
private:
};

#define LOG_OUT(info) std::cout << info; out_log << info;

//////////////////////////////////////////////////////////////////////////

void get_all_files(const std::string& root, std::vector<std::string>&file_paths , const std::set< std::string >&postfix)
{
    if (root.empty())
    {
        return;
    }
    else
    {
        std::vector<std::string> dirs;
        for (boost::filesystem::directory_iterator it(root) ; it != boost::filesystem::directory_iterator() ; ++it)
        {
            if (boost::filesystem::is_directory(*it))
            {
                dirs.push_back((*it).path().filename().string());
            }
            else
            {
                const std::string ext = boost::filesystem::extension(*it);
                if (postfix.find(ext) != postfix.end())
                {
                    file_paths.push_back(root + "/" + (*it).path().filename().string());
                }
            }
        }
    }
}

int load_image(const std::string& file, MultiResolutionImage*& img)
{
    img = MultiResolutionImageFactory::openImage(file);
    if (img)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int load_anno(const std::string& file , std::shared_ptr<AnnotationList>& anno_list)
{
    anno_list.reset(new AnnotationList());
    BinaryRepository rep(anno_list);
    rep.setSource(file);
    if (rep.load())
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int get_proper_image(MultiResolutionImage* img , 
    int &pp_level , int (&pp_dim)[2] , double& pp_ratio , unsigned char* &pp_img_gray)
{
    const double pp_size = 800;
    pp_level = 0;
    double dis = std::numeric_limits<double>::max();
    for (int level = 0; level< img->getNumberOfLevels() ; ++level)
    {
        std::vector<unsigned long long> dims = img->getLevelDimensions(level);
        unsigned long long max_dim = (std::max)(dims[0], dims[1]);
        const double cur_dis = fabs(static_cast<double>(max_dim) - static_cast<double>(pp_size));
        if (cur_dis < dis)
        {
            dis = cur_dis;
            pp_level = level;
            pp_dim[0] = static_cast<int>(dims[0]);
            pp_dim[1] = static_cast<int>(dims[1]);
        }
    }

    pp_ratio = img->getLevelDownsample(pp_level);

    if ((std::max)(pp_dim[0], pp_dim[1]) > 4096)
    {
        LOG_OUT("WARNING : get proper level image dim is larget than 4096!");
    }

    unsigned char* pp_img_rgb = nullptr;
    img->getRawRegion(0, 0 , pp_dim[0], pp_dim[1], pp_level, pp_img_rgb);
    if (nullptr == pp_img_rgb)
    {
        LOG_OUT("ERROR : get proper level image failed!");
        return -1;
    }

    if (pp_img_gray != nullptr)
    {
        delete[] pp_img_gray;
        pp_img_gray = nullptr;
    }
    pp_img_gray = new unsigned char[pp_dim[0] * pp_dim[1]];
    for (int i = 0 ; i< pp_dim[0]*pp_dim[1] ; ++ i)
    {
        //0.2989 * R + 0.5870 * G + 0.1140 * B 
        double gray = static_cast<double>(pp_img_rgb[i * 3]) *0.2989 +
            static_cast<double>(pp_img_rgb[i * 3 + 1])*0.5870 +
            static_cast<double>(pp_img_rgb[i * 3 + 2])*0.1140;
        gray = gray > 255.0 ? 255.0 : gray;
        pp_img_gray[i] = static_cast<unsigned char>(gray);
    }

    //TODO DEBUG
    write_raw("D:/temp/pimg.rgb", (char*)pp_img_rgb, pp_dim[0] * pp_dim[1] * 3);
    write_raw("D:/temp/pimg.gray", (char*)pp_img_gray, pp_dim[0] * pp_dim[1]);

    delete[] pp_img_rgb;
    pp_img_rgb = nullptr;
    return 0;
}

AABB<int> get_anno_aabb(std::shared_ptr<Annotation> anno, double ratio)
{
    std::vector<Point> img_bounding = anno->getImageBoundingBox();

    AABB<int> aabb;
    aabb._min[0] = static_cast<int>(img_bounding[0].getX() / ratio);
    aabb._min[1] = static_cast<int>(img_bounding[0].getY() / ratio);
    aabb._max[0] = static_cast<int>(img_bounding[1].getX() / ratio);
    aabb._max[1] = static_cast<int>(img_bounding[1].getY() / ratio);

    return aabb;
}

std::vector <AABB<int>> combine_aabb(const std::vector<AABB<int>>& aabbs , int sensitive_border)
{
    std::vector<AABB<int>> c_aabbs = aabbs;

    bool no_combine = false;
    while (!no_combine)
    {
        no_combine = true;

        for (int i = 0 ; i < (int)c_aabbs.size() ; ++i)
        {
            for (int j = i +1 ; j < (int)c_aabbs.size(); ++j)
            {
                AABB<int> com_aabb;
                if (aabb_to_aabb_combine<int>(c_aabbs[i], c_aabbs[j], com_aabb))
                {
                    std::vector<AABB<int>> new_aabbs;
                    new_aabbs.push_back(com_aabb);
                    for (int k= 0 ; k < (int)c_aabbs.size() ; ++k)
                    {
                        if (k == i || k== j)
                        {
                            continue;
                        }
                        new_aabbs.push_back(c_aabbs[k]);
                    }
                    std::swap(c_aabbs, new_aabbs);
                    no_combine = false;
                    goto HAS_COMBINE;
                }
                else
                {
                    float c0[2] = {
                        (c_aabbs[i]._min[0] + c_aabbs[i]._max[0])*0.5f,
                        (c_aabbs[i]._min[1] + c_aabbs[i]._max[1])*0.5f};
                    float c1[2] ={
                        (c_aabbs[j]._min[0] + c_aabbs[j]._max[0])*0.5f,
                        (c_aabbs[j]._min[1] + c_aabbs[j]._max[1])*0.5f };
                    const float dis = sqrt((c0[0] - c1[0])*(c0[0] - c1[0]) + (c0[1] - c1[1])*(c0[1] - c1[1]));
                    if (dis < (float)sensitive_border)
                    {
                        for (int q = 0; q< 2; ++q)
                        {
                            com_aabb._max[q] = (std::max)(c_aabbs[i]._max[q], c_aabbs[j]._max[q]);
                            com_aabb._min[q] = (std::min)(c_aabbs[i]._min[q], c_aabbs[j]._min[q]);
                        }

                        std::vector<AABB<int>> new_aabbs;
                        new_aabbs.push_back(com_aabb);
                        for (int k = 0; k < (int)c_aabbs.size(); ++k)
                        {
                            if (k == i || k == j)
                            {
                                continue;
                            }
                            new_aabbs.push_back(c_aabbs[k]);
                        }
                        std::swap(c_aabbs, new_aabbs);
                        no_combine = false;
                        goto HAS_COMBINE;
                    }
                }
                
            }
        }

    HAS_COMBINE:;

    }

    return c_aabbs;
}

std::vector <AABB<int>> expand_aabb(const std::vector<AABB<int>>& aabbs, int (&dim)[2] , double expand_ratio)
{
    std::vector<AABB<int>> ex_aabbs;
    ex_aabbs.reserve(aabbs.size());
    for (size_t i = 0 ; i< aabbs.size() ; ++i)
    {
        const AABB<int>& aabb = aabbs[i];
        double c0[2] = {
            (aabbs[i]._min[0] + aabbs[i]._max[0])*0.5,
            (aabbs[i]._min[1] + aabbs[i]._max[1])*0.5 };
        double w = aabbs[i]._max[0] - aabbs[i]._min[0];
        double h = aabbs[i]._max[1] - aabbs[i]._min[1];
        w *= expand_ratio;
        h *= expand_ratio;

        AABB<int> e_aabb;
        e_aabb._min[0] = (int)c0[0] - (int)(w*0.5);
        e_aabb._max[0] = (int)c0[0] + (int)(w*0.5);
        e_aabb._min[1] = (int)c0[1] - (int)(h*0.5);
        e_aabb._max[1] = (int)c0[1] + (int)(h*0.5);

        e_aabb._min[0] = e_aabb._min[0] < 0 ? 0 : e_aabb._min[0];
        e_aabb._min[1] = e_aabb._min[1] < 0 ? 0 : e_aabb._min[1];

        e_aabb._max[0] = e_aabb._max[0] > dim[0]-1 ? dim[0] - 1 : e_aabb._max[0];
        e_aabb._max[1] = e_aabb._max[1] > dim[1] - 1 ? dim[1] - 1 : e_aabb._max[1];

        ex_aabbs.push_back(e_aabb);
    }

    return ex_aabbs;
}

std::vector<AABB<int>> get_image_connected_domina_aabb(int(&dim)[2] , unsigned char* img_gray , std::vector<AABB<int>>c_aabb )
{
    //1 get mask
    SegmentAnalysis seg;
    double th= 0;
    for (auto it = c_aabb.begin() ; it != c_aabb.end() ; ++it )
    {
        th += (double)(seg.get_threshold_otus_low(img_gray, dim[0], dim[1], *it));
    }
    th = th / (double)(c_aabb.size());
    if (th > 255)
    {
        //ERROR
    }
    std::cout << "thresold is : " << th << std::endl;

    unsigned char* mask = new unsigned char[dim[0] * dim[1]];
    seg.segment_low(img_gray, mask, dim[0], dim[1] , (unsigned char)th);

    Morphology mor;
    mor.erose(mask, dim[0], dim[1], 1);
    mor.dilate(mask, dim[0], dim[1], 1);

    mor.erose(mask, dim[0], dim[1], 1);
    mor.dilate(mask, dim[0], dim[1], 1);

    mor.erose(mask, dim[0], dim[1], 1);
    mor.dilate(mask, dim[0], dim[1], 1);

    //mor.dilate(mask, dim[0], dim[1], 3);

    //2 get connected_domain
    ConnectedDomainAnalysis cd;
    cd.set_dim(dim);
    cd.set_mask_ref(mask);

    int roi_min[2] = { 0,0 };
    int roi_max[2] = {dim[0],dim[1]};
    cd.set_roi(roi_min, roi_max);
    std::vector<AABB<int>> aabb = cd.get_connected_domain_aabb(50);

    //TODO DEBUG
    //write_raw("D:/temp/pmask.raw", (char*)mask, dim[0] * dim[1]);


    delete[] mask;
    return aabb;
}
//
//std::vector<AABB<int>> shrink_aabb(const std::vector<AABB<int>>& aabbs, int (&dim)[2] , unsigned char* img_gray)
//{
//    unsigned int ;
//    for (auto it_aabb = aabbs.begin() ;it_aabb != aabbs.end() ; ++it_aabb)
//    {
//        const AABB<int>& aabb = *it_aabb;
//        std::unique_ptr<unsigned char> img_region;
//
//    }
//}



void get_vis_info(int(&pp_dim)[2], double pp_ratio, unsigned char* pp_img_gray,
    const std::vector<std::shared_ptr<Annotation>>& annos,
    const std::vector<AABB<int>>& aabbs,
    const std::vector<AABB<int>>& c_aabbs,
    const std::vector<AABB<int>>& e_aabbs)
{
    _img_width = pp_dim[0];
    _img_height = pp_dim[1];
    _pp_ratio = pp_ratio;
    _vis_annos = annos;
    _vis_aabbs = aabbs;
    _vis_c_aabbs = c_aabbs;
    _vis_e_aabbs = e_aabbs;

    if (nullptr != _vis_img_buffer)
    {
        delete[] _vis_img_buffer;
        _vis_img_buffer = nullptr;
    }

    const int img_size = _img_width * _img_height;
    _vis_img_buffer = new unsigned char[img_size * 3];
    for (int i = 0; i <img_size; ++i)
    {
        _vis_img_buffer[i * 3] = pp_img_gray[i];
        _vis_img_buffer[i * 3 + 1] = pp_img_gray[i];
        _vis_img_buffer[i * 3 + 2] = pp_img_gray[i];
    }
}

int process_one_anno(const std::string& anno_file, const std::string& img_file,  const std::string& output_dir , int flag, int sensitive_border , double expand_ratio)
{
    MultiResolutionImage* img = nullptr;
    if (0 != load_image(img_file, img))
    {
        LOG_OUT("ERROR : image file : " + img_file + std::string(" load failed!"));
        return -1;
    }

    std::shared_ptr<AnnotationList> anno_list;
    if (0 != load_anno(anno_file , anno_list))
    {
        LOG_OUT("ERROR : annotation file : " + anno_file+ std::string(" load failed!"));
        return -1;
    }
    std::vector<std::shared_ptr<Annotation>> annos = anno_list->getAnnotations();

    int pp_level = 0;
    double pp_ratio = 0;
    int pp_dim[2] = { 0,0 };
    unsigned char* pp_img_gray = nullptr;
    get_proper_image(img, pp_level, pp_dim, pp_ratio, pp_img_gray);

    std::vector<AABB<int>> aabbs;
    for (int i = 0; i<annos.size() ; ++i)
    {
        aabbs.push_back(get_anno_aabb(annos[i], pp_ratio));
    }
    std::vector<AABB<int>> c_aabbs = combine_aabb(aabbs , sensitive_border);

    std::vector<AABB<int>> expand_aabbs = expand_aabb(c_aabbs, pp_dim, expand_ratio);

    std::vector<AABB<int>> img_cd_aabb = get_image_connected_domina_aabb(pp_dim, pp_img_gray , c_aabbs);

    get_vis_info(pp_dim, pp_ratio , pp_img_gray ,annos , aabbs , c_aabbs , expand_aabbs);


    delete[] pp_img_gray;

    return 0;
}

int anno_to_mask_image(
    const std::vector<std::string>& anno_files,
    const std::vector<std::string>& img_files,
    const std::string& output_dir,
    int flag, 
    int sensitive_border,
    double expand_ratio)
{
    for (auto it = anno_files.begin(); it != anno_files.end(); ++it)
    {
        const std::string& anno_file = *it;
        const std::string anno_base =  boost::filesystem::basename(anno_file);
        std::string img_file;
        for (auto it2 = img_files.begin(); it2 != img_files.end() ; ++it2)
        {
            const std::string img_base = boost::filesystem::basename(*it2);
            if (img_base == anno_base)
            {
                img_file = *it2;
                break;
            }
        }

        if (!img_file.empty())
        {
            if (0 != process_one_anno(anno_file, img_file , output_dir , flag, sensitive_border ,expand_ratio))
            {
                LOG_OUT("ERROR : annotation file : " + anno_file + std::string(" processing failed!"));
                continue;
            }
        }
        else
        {
            LOG_OUT("WARNING : annotation file : " + anno_file + std::string(" match image failed!"));
            continue;
        }
    }


    return 0;
}


unsigned int _fbo = 0;
unsigned int _tex = 0;
unsigned int _depth = 0;
void display()
{
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _fbo);
    glViewport(0, 0, _img_width, _img_height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawPixels(_img_width, _img_height, GL_RGB, GL_UNSIGNED_BYTE, _vis_img_buffer);

    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, _img_width, 0, _img_height);

    glLineWidth(5.0);

    glColor3d(0.0, 0.2, 0.9);
    for (auto it = _vis_annos.begin() ; it != _vis_annos.end() ; ++it)
    {
        std::shared_ptr<Annotation> anno = *it;
        const std::vector<Point>& pts =  anno->getCoordinates();
        glBegin(GL_LINE_STRIP);
        for (auto it2 = pts.begin() ; it2 != pts.end() ; ++it2)
        {
            double x = (*it2).getX() / _pp_ratio;
            double y = (*it2).getY() / _pp_ratio;
            glVertex2d(x, y);
        }
        glEnd();
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3d(0.9, 0.2, 0.2);
    for (auto it = _vis_aabbs.begin(); it != _vis_aabbs.end(); ++it)
    {
        const AABB<int> aabb = *it;
        glBegin(GL_QUADS);
        glVertex2i(aabb._min[0], aabb._min[1]);
        glVertex2i(aabb._max[0], aabb._min[1]);
        glVertex2i(aabb._max[0], aabb._max[1]);
        glVertex2i(aabb._min[0], aabb._max[1]);
        glEnd();
    }

    glLineWidth(7.0);
    glColor3d(0.2, 0.9, 0.2);
    for (auto it = _vis_c_aabbs.begin(); it != _vis_c_aabbs.end(); ++it)
    {
        const AABB<int> aabb = *it;
        glBegin(GL_QUADS);
        glVertex2i(aabb._min[0], aabb._min[1]);
        glVertex2i(aabb._max[0], aabb._min[1]);
        glVertex2i(aabb._max[0], aabb._max[1]);
        glVertex2i(aabb._min[0], aabb._max[1]);
        glEnd();
    }

    glLineWidth(8.0);
    glColor3d(0.8, 0.9, 0.2);
    for (auto it = _vis_e_aabbs.begin(); it != _vis_e_aabbs.end(); ++it)
    {
        const AABB<int> aabb = *it;
        glBegin(GL_QUADS);
        glVertex2i(aabb._min[0], aabb._min[1]);
        glVertex2i(aabb._max[0], aabb._min[1]);
        glVertex2i(aabb._max[0], aabb._max[1]);
        glVertex2i(aabb._min[0], aabb._max[1]);
        glEnd();
    }

    glPopMatrix();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glDrawBuffer(GL_BACK);
    glBlitFramebuffer(0, 0, _img_width, _img_height, 0, _height, _width, 0 , GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glutSwapBuffers();
}

void reshape(int width , int height)
{
    if (width == 0 || height == 0)
    {
        return;
    }

    _width = width;
    _height = height;
    glutPostRedisplay();
}

void init()
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glGenFramebuffers(1, &_fbo);
    glGenTextures(1, &_tex);
    glGenTextures(1, &_depth);

    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    glBindTexture(GL_TEXTURE_2D, _tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _img_width, _img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _tex, 0);

    glBindTexture(GL_TEXTURE_2D, _depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, _img_width, _img_height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depth, 0);

}


int main(int argc, char* argv[])
{
    LogSheild log_sheild("anno2mask.log", "Extracting mask/image from annotation files>>> \n");

    /*std::string image_direction;
    std::string annotation_direction;
    std::string output_direction;
    bool encryption = true;*/

    std::string image_dir = "E:/data/Pathology/Script/Test0";
    std::string annotation_dir = image_dir;
    std::string output_dir = image_dir;
    int sensitive_border = 200;
    double expand_ratio = 2.5;

    bool encryption = true;


    std::set<std::string> anno_post_fix;
    anno_post_fix.insert(".araw");
    std::vector<std::string> anno_files;
    get_all_files(annotation_dir, anno_files, anno_post_fix);
    if(anno_files.empty())
    {
        LOG_OUT("ERROR : annotation file is empty !")
    }

    std::set<std::string> img_post_fix;
    img_post_fix.insert(".tif");
    img_post_fix.insert(".svs");
    img_post_fix.insert(".vms");
    img_post_fix.insert(".vmu");
    img_post_fix.insert(".ndpi");
    img_post_fix.insert(".scn");
    img_post_fix.insert(".mrxs");
    img_post_fix.insert(".tiff");
    img_post_fix.insert(".svslide");
    img_post_fix.insert(".bif");
    std::vector<std::string> img_files;
    get_all_files(image_dir, img_files, img_post_fix);
    if (img_files.empty())
    {
        LOG_OUT("ERROR : image file is empty!");
    }

    if (0 != anno_to_mask_image(anno_files ,img_files, output_dir , 0, sensitive_border , expand_ratio))
    {
        LOG_OUT("ERROR : annotation to mask image faild!");
    }

    //////////////////////////////////////////////////////////////////////////
    //VIS part
    //////////////////////////////////////////////////////////////////////////

    if (_img_width > 1080 || _img_height > 1080)
    {
        _width = 1080;
        _height = (int)((double)_width / (double)_img_width * (double)_img_height);
    }
    else
    {
        _width = _img_width;
        _height = _img_height;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(_width, _height);

    glutCreateWindow("Annotation-Vis");

    if (GLEW_OK != glewInit())
    {
        std::cout << "Init glew failed!\n";
        return -1;
    }

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    //glutKeyboardFunc(keybord);

    glutMainLoop();

    return 0;
}
