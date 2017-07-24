#include <string>
#include <fstream>
#include <vector>

#include "MedImgCommon/mi_common_file_util.h"
#include "MedImgArithmetic/mi_scan_line_analysis.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgCommon/mi_string_number_converter.h"
#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_image_data.h"
#include "Ext/pugixml/pugixml.hpp"
#include "Ext/pugixml/pugiconfig.hpp"

using namespace medical_imaging;

const std::string path_direction = "E:/data/LIDC/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/";
const std::string xml_file = path_direction + std::string("/069.xml");
const bool save_slice_location_less = false;

struct Nodule
{
    int type;//0 for unblinded-nodule ; 1 for non-nodule
    std::string name;
    std::vector<Point3> _points;
};

std::vector<std::string> get_dicom_files()
{
    std::vector<std::string> files;
    FileUtil::get_all_file_recursion(path_direction , std::vector<std::string>(1,".dcm") , files);
    return files;
}

bool load_dicom_series(std::vector<std::string>& files ,
    std::shared_ptr<ImageDataHeader>& header,
    std::shared_ptr<ImageData>& img)
{
    DICOMLoader loader;
    IOStatus status = loader.load_series(files , img , header);
    return status == IO_SUCCESS;
}

bool point_less(const Point3& l, const Point3& r)
{
    return l.z < r.z;
}

void get_nodule_set(const std::string& xml_file);


int main(int argc , char* argv[])
{
    pugi::xml_document doc;
    if (!doc.load_file(xml_file.c_str()))
    {
        std::cout << "Load file failed!\n!";
        return -1;
    }

    //Root
    pugi::xml_node root_node = doc.child("LidcReadMessage");
    if (root_node.empty())
    {
        std::cout << "invalid format , find Lidc read message failed!\n";
        return -1;
    }

    //Response header
    pugi::xml_node response_header_node = root_node.child("ResponseHeader");
    if (response_header_node.empty())
    {
        std::cout << "invalid format , find Lidc response header failed!\n";
        return -1;
    }

    //Study UID
    pugi::xml_node study_uid_node = response_header_node.child("StudyInstanceUID");
    if (study_uid_node.empty())
    {
        std::cout << "invalid format , find study UID failed!\n";
        return -1;
    }
    const std::string study_uid = study_uid_node.child_value();
    std::cout << "study uid : " << study_uid << std::endl;

    //Series UID
    pugi::xml_node series_uid_node = response_header_node.child("SeriesInstanceUid");
    if (series_uid_node.empty())
    {
        std::cout << "invalid format , find series UID failed!\n";
        return -1;
    }
    const std::string series_uid = series_uid_node.child_value();
    std::cout << "series uid : " << series_uid << std::endl;


    std::vector<Nodule> _nodules;

    StrNumConverter<double> str_to_num;
    pugi::xpath_node_set reading_session_node_set = root_node.select_nodes("readingSession");
    for (auto it = reading_session_node_set.begin() ; it != reading_session_node_set.end(); ++it)
    {
        pugi::xpath_node_set unblind_nodule_node_set = (*it).node().select_nodes("unblindedReadNodule");
        for (auto it2 = unblind_nodule_node_set.begin() ; it2 != unblind_nodule_node_set.end(); ++it2)
        {
            pugi::xpath_node node_unblinded = (*it2);
            Nodule nodule;
            nodule.type = 0;

            //get ID
            pugi::xml_node id_node = node_unblinded.node().child("noduleID");
            if (id_node.empty())
            {
                //TODO ERROR
                std::cout << "invalid format , find nodule ID failed!\n";
                return -1;
            }
            nodule.name = id_node.child_value();

            //get characteristics TODO

            //get contour
            pugi::xpath_node_set roi_node_set = node_unblinded.node().select_nodes("roi");
            for (auto it3 = roi_node_set.begin() ; it3 != roi_node_set.end(); ++it3)
            {
                pugi::xpath_node roi_node = (*it3);
                pugi::xml_node pos_z_node = roi_node.node().child("imageZposition");
                if (pos_z_node.empty())
                {
                    //TODO ERROR
                    std::cout << "invalid format , find image position z failed!\n";
                    return -1;
                }
                const double pos_z = str_to_num.to_num(pos_z_node.child_value());

                pugi::xpath_node_set edge_node_set = roi_node.node().select_nodes("edgeMap");
                for (auto it4 = edge_node_set.begin() ; it4 != edge_node_set.end(); ++it4)
                {
                    pugi::xpath_node edge_node = (*it4);
                    pugi::xml_node x_node = edge_node.node().child("xCoord");
                    if (x_node.empty())
                    {
                        //TODO ERROR
                        std::cout << "invalid format , find edge position x failed!\n";
                        return -1;
                    }
                    pugi::xml_node y_node = edge_node.node().child("yCoord");
                    if (y_node.empty())
                    {
                        //TODO ERROR
                        std::cout << "invalid format , find edge position y failed!\n";
                        return -1;
                    }
                    const double pos_x = str_to_num.to_num(x_node.child_value());
                    const double pos_y = str_to_num.to_num(y_node.child_value());
                    nodule._points.push_back(Point3(pos_x , pos_y , pos_z));
                }

            }

            _nodules.push_back(nodule);
        }
    }

    //slice location to pixal coordinate
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    if(!load_dicom_series(get_dicom_files() ,_data_header,_volume_data ))
    {
        std::cout << "load series failed!\n";
        return -1;
    }
    std::vector<double> slice_location = _data_header->slice_location;
    if (!save_slice_location_less)
    {
        std::sort(slice_location.begin() , slice_location.end() , std::greater<double>());//slice location from max to min
    }

    double delta = 0;
    for(int i = 1; i<slice_location.size() ; ++i)
    {
        if (fabs(slice_location[i] - slice_location[i-1]) > DOUBLE_EPSILON)
        {
            delta = slice_location[i] - slice_location[i-1];
        }
    }

    if (fabs(delta) < DOUBLE_EPSILON )
    {
        std::cout << "slice location error!\n";
        return -1;
    }

    const double slice0 = slice_location[0];

    for (auto it = _nodules.begin() ; it != _nodules.end() ; ++it)
    {
        Nodule& nodule = *it;
        std::vector<Point3>& pts = nodule._points;
        for (int  i= 0 ; i< pts.size() ; ++i)
        {
            double slice = pts[i].z;
            double delta_slice = slice - slice0;
            int tmp_idx = static_cast<int>(delta_slice / delta);
            if (tmp_idx > slice_location.size())
            {
                std::cout << "find slice lotation failed!\n";
                return -1;
            }

            if (fabs(slice_location[tmp_idx] - slice) < DOUBLE_EPSILON )
            {
                pts[i].z = static_cast<double>(tmp_idx);
                goto FIND_LOCATION;
            }
            else if (slice_location[tmp_idx] - slice < 0)
            {
                if (save_slice_location_less)
                {
                    for(int j = tmp_idx ; j<slice_location.size() ; ++j)
                    {
                        if (fabs(slice_location[j] - slice) < DOUBLE_EPSILON )
                        {
                            pts[i].z = static_cast<double>(j);
                            goto FIND_LOCATION;
                        }
                    }
                }
                else
                {
                    for(int j = tmp_idx ; j>0 ; --j)
                    {
                        if (fabs(slice_location[j] - slice) < DOUBLE_EPSILON )
                        {
                            pts[i].z = static_cast<double>(j);
                            goto FIND_LOCATION;
                        }
                    }
                }
                

                std::cout << "find slice lotation failed!\n";
                return -1;
            }
            else
            {
                if (save_slice_location_less)
                {
                    for(int j = tmp_idx ; j>0 ; --j)
                    {
                        if (fabs(slice_location[j] - slice) < DOUBLE_EPSILON )
                        {
                            pts[i].z = static_cast<double>(j);
                            goto FIND_LOCATION;
                        }
                    }
                }
                else
                {
                    for(int j = tmp_idx ; j<slice_location.size() ; ++j)
                    {
                        if (fabs(slice_location[j] - slice) < DOUBLE_EPSILON )
                        {
                            pts[i].z = static_cast<double>(j);
                            goto FIND_LOCATION;
                        }
                    }
                }
                

                std::cout << "find slice lotation failed!\n";
                return -1;
            }

FIND_LOCATION:;
        }
    }

    //contour to mask
    std::shared_ptr<ImageData> _mask(new ImageData);
    _volume_data->shallow_copy(_mask.get());
    _mask->_data_type = UCHAR;
    _mask->mem_allocate();

    ScanLineAnalysis<unsigned char> scan_line_analysis;
    typedef ScanLineAnalysis<unsigned char>::Pt2 PT2;
    unsigned char label = 0;
    for (auto it = _nodules.begin() ; it != _nodules.end() ; ++it)
    {
        Nodule& nodule = *it;
        std::vector<Point3>& pts = nodule._points;
        if (pts.size() < 10)//TODO skip some to test
        {
            continue;
        }

        ++label;

        std::sort(pts.begin() , pts.end() , point_less);

        bool begin = true;
        int current_z = 0;
        std::vector<PT2> current_contour;
        
        for (int i = 0 ; i < pts.size() ; ++i)
        {
            if(begin)
            {
                current_z = static_cast<int>(pts[i].z);
                current_contour.push_back(PT2(static_cast<int>(pts[i].x) , static_cast<int>(pts[i].y) ) );
                begin = false;
            }
            else
            {
                int z = static_cast<int>(pts[i].z);
                if (z == current_z)//push contour
                {
                    current_contour.push_back(PT2(static_cast<int>(pts[i].x) , static_cast<int>(pts[i].y) ) );
                }
                else// do scaning
                {
                    scan_line_analysis.fill((unsigned char*)_mask->get_pixel_pointer() + current_z*_mask->_dim[0]*_mask->_dim[1] ,
                        _mask->_dim[0] , _mask->_dim[1] , current_contour , label);


                    //back to begin
                    --i;
                    current_contour.clear();
                    begin = true;
                }
            }
            
        }
    }


    FileUtil::write_raw(path_direction+std::string("/mask.raw") , _mask->get_pixel_pointer() , _mask->get_data_size());
    //FileUtil::write_raw(path_direction+std::string("/volume.raw") , _volume_data->get_pixel_pointer() , _volume_data->get_data_size());





    std::cout << "Done\n";
    return 0;
}