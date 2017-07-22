#include <string>
#include <fstream>
#include <vector>

#include "MedImgCommon/mi_common_file_util.h"
#include "MedImgArithmetic/mi_scan_line_analysis.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgCommon/mi_string_number_converter.h"
#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data_header.h"
#include "Ext/pugixml/pugixml.hpp"
#include "Ext/pugixml/pugiconfig.hpp"

using namespace medical_imaging;

const std::string path_direction = "E:/data/LIDC/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/";
const std::string xml_file = path_direction + std::string("/069.xml");

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

void load_dicom_series(std::vector<std::string>& files ,
    std::shared_ptr<ImageDataHeader>& header,
    std::shared_ptr<ImageData>& img)
{
    DICOMLoader loader;
    IOStatus status = loader.load_series(files , img , header);
}

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
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    load_dicom_series(get_dicom_files() ,_data_header,_volume_data );

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




    std::cout << "Done\n";
    return 0;
}