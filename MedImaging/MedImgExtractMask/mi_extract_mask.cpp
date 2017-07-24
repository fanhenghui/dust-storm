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
#include "MedImgIO/mi_run_length_operator.h"
#include "Ext/pugixml/pugixml.hpp"
#include "Ext/pugixml/pugiconfig.hpp"

using namespace medical_imaging;

//////////////////////////////////////////////////////////////////////////
static std::ofstream out_log;
class LogSheild
{
public:
    LogSheild(const std::string& log_file , const std::string& start_word )
    {
        out_log.open("anon.log" , std::ios::out);
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

struct Nodule
{
    int type;//0 for unblinded-nodule ; 1 for non-nodule
    std::string name;
    std::vector<Point3> _points;
};

bool point_less(const Point3& l, const Point3& r)
{
    return l.z < r.z;
}

int load_dicom_series(std::vector<std::string>& files ,
    std::shared_ptr<ImageDataHeader>& header,
    std::shared_ptr<ImageData>& img)
{
    DICOMLoader loader;
    if(loader.load_series(files , img , header) == IO_SUCCESS)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int get_nodule_set(const std::string& xml_file, std::vector<Nodule>& nodules , std::string& series_uid)
{
    pugi::xml_document doc;
    if (!doc.load_file(xml_file.c_str()))
    {
       LOG_OUT("Load file failed!\n!");
        return -1;
    }

    //Root
    pugi::xml_node root_node = doc.child("LidcReadMessage");
    if (root_node.empty())
    {
       LOG_OUT( "invalid format , find Lidc read message failed!\n");
        return -1;
    }

    //Response header
    pugi::xml_node response_header_node = root_node.child("ResponseHeader");
    if (response_header_node.empty())
    {
       LOG_OUT("invalid format , find Lidc response header failed!\n");
        return -1;
    }

    //Study UID
    pugi::xml_node study_uid_node = response_header_node.child("StudyInstanceUID");
    if (study_uid_node.empty())
    {
        LOG_OUT("invalid format , find study UID failed!\n");
        return -1;
    }
    const std::string study_uid = study_uid_node.child_value();
    //LOG_OUT( std::string("study uid : ") + study_uid + "\n");

    //Series UID
    pugi::xml_node series_uid_node = response_header_node.child("SeriesInstanceUid");
    if (series_uid_node.empty())
    {
        LOG_OUT( "invalid format , find series UID failed!\n");
        return -1;
    }
    series_uid = series_uid_node.child_value();
    //LOG_OUT( "series uid : " + series_uid + "\n");
    if (series_uid.empty())
    {
        LOG_OUT( "invalid format , empty series UID !\n");
        return -1;
    }

    

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
                LOG_OUT(  "invalid format , find nodule ID failed!\n");
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
                    LOG_OUT("invalid format , find image position z failed!\n");
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
                        LOG_OUT( "invalid format , find edge position x failed!\n");
                        return -1;
                    }
                    pugi::xml_node y_node = edge_node.node().child("yCoord");
                    if (y_node.empty())
                    {
                        //TODO ERROR
                        LOG_OUT( "invalid format , find edge position y failed!\n");
                        return -1;
                    }
                    const double pos_x = str_to_num.to_num(x_node.child_value());
                    const double pos_y = str_to_num.to_num(y_node.child_value());
                    nodule._points.push_back(Point3(pos_x , pos_y , pos_z));
                }

            }

            nodules.push_back(nodule);
        }
    }

    return 0;
}

int resample_z(std::vector<Nodule>& nodules , std::shared_ptr<ImageDataHeader>& header , bool save_slice_location_less)
{
    std::vector<double> slice_location = header->slice_location;
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
        LOG_OUT(  "slice location error!\n");
        return -1;
    }

    const double slice0 = slice_location[0];

    for (auto it = nodules.begin() ; it != nodules.end() ; ++it)
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
                LOG_OUT( "find slice lotation failed!\n");
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

                LOG_OUT(  "find slice lotation failed!\n");
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


                LOG_OUT( "find slice lotation failed!\n");
                return -1;
            }

FIND_LOCATION:;
        }
    }

    return 0;
}

int contour_to_mask(std::vector<Nodule>& nodules , std::shared_ptr<ImageData> mask)
{
    mask->_data_type = UCHAR;
    mask->mem_allocate();

    ScanLineAnalysis<unsigned char> scan_line_analysis;
    typedef ScanLineAnalysis<unsigned char>::Pt2 PT2;
    unsigned char label = 0;
    for (auto it = nodules.begin() ; it != nodules.end() ; ++it)
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
                    scan_line_analysis.fill((unsigned char*)mask->get_pixel_pointer() + current_z*mask->_dim[0]*mask->_dim[1] ,
                        mask->_dim[0] , mask->_dim[1] , current_contour , label);

                    //back to begin
                    --i;
                    current_contour.clear();
                    begin = true;
                }
            }
        }
    }

    return 0;
}

int save_mask(std::shared_ptr<ImageData> mask , const std::string& path , bool compressed)
{
    if (path.empty())
    {
        LOG_OUT("save mask path is empty!");
        return -1;
    }
    if (compressed)
    {
        std::vector<unsigned int> code = RunLengthOperator::encode((unsigned char*)mask->get_pixel_pointer() , mask->get_data_size());
        if (code.empty())
        {
            LOG_OUT("mask is all zero!");
            return -1;
        }
        return FileUtil::write_raw(path , code.data() , static_cast<unsigned int>(code.size())*sizeof(unsigned int));
    }
    else
    {
        return FileUtil::write_raw(path , mask->get_pixel_pointer() , mask->get_data_size());
    }
}

int browse_root_xml(const std::string& root , std::vector<std::string>& xml_files )
{
    if (root.empty()) {
        LOG_OUT("empty root!");
        return -1;
    }

    std::map<std::string , std::vector<std::string> >files;
    std::set<std::string > postfix;
    postfix.insert(".xml");

    FileUtil::get_all_file_recursion(root , postfix , files);
    if (files.empty())
    {
        LOG_OUT("empty files!")
        return -1;
    }

    if (files.find(".xml") != files.end() )
    {
        xml_files = files[".xml"];
    }
    return 0;
}

int browse_root_dcm(const std::string& root, std::map<std::string, std::vector<std::string>>& dcm_files )
{
    if (root.empty()) {
        LOG_OUT("empty root!");
        return -1;
    }

    std::map<std::string , std::vector<std::string> >files;
    std::set<std::string > postfix;
    postfix.insert(".dcm");

    FileUtil::get_all_file_recursion(root , postfix , files);
    if (files.empty())
    {
        LOG_OUT("empty files!")
            return -1;
    }

    if (files.find(".dcm") != files.end())
    {
        DICOMLoader loader;
        std::vector<std::string>& all_dcm_files = files.find(".dcm")->second;
        for (auto it = all_dcm_files.begin() ; it != all_dcm_files.end() ; ++it)
        {
            std::string study_uid;
            std::string series_uid;
            loader.check_series_uid(*it , study_uid , series_uid);
            if (dcm_files.find(series_uid) == dcm_files.end())
            {
                std::vector<std::string> sub_dcm_files(1 , *it);
                dcm_files[series_uid] = sub_dcm_files;
            }
            else
            {
                dcm_files[series_uid].push_back(*it);
            }
        }
    }
    return 0;
}


int main(int argc , char* argv[])
{
    /*arguments list:
    -help : print all argument
    -data <path> : dicom data root(.dcm)
    -annotation <path> : annotation root(.xml)
    -output <path] : save mask root
    -compress : if mask is compressed 
    -slicelocation <less/greater> : default is less
    */

    LogSheild log_sheild("em.log" , "Extracting mask from LIDC data set >>> \n");

    std::string dcm_direction;
    std::string annotation_direction;
    std::string output_direction;
    bool compressed = false;
    bool save_slice_location_less = true;

    if (argc == 1)
    {
        LOG_OUT("invalid arguments!\n");
        LOG_OUT("targuments list:\n");
        LOG_OUT("\t-annotation <path> : annotation root(.dcm)\n");
        LOG_OUT("\t-annotation <path> : annotation root(.xml)\n");
        LOG_OUT("\t-output <path] : save mask root\n");
        LOG_OUT("\t-compress : if mask is compressed\n");
        LOG_OUT("\t-slicelocation <less/greater> : default is less\n");
        return -1;
    }
    else
    {
       for (int i = 1; i< argc ; ++i)
       {
           if (std::string(argv[i]) == "-help")
           {
               LOG_OUT("arguments list:\n");
               LOG_OUT("\t-annotation <path> : annotation root(.dcm)\n");
               LOG_OUT("\t-annotation <path> : annotation root(.xml)\n");
               LOG_OUT("\t-output <path] : save mask root\n");
               LOG_OUT("\t-compress : if mask is compressed\n");
               LOG_OUT("\t-slicelocation <less/greater> : default is less\n");
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
        std::vector<Nodule> nodules;
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

        //contour to mask
        std::shared_ptr<ImageData> mask(new ImageData);
        volume_data->shallow_copy(mask.get());
        if (0 != contour_to_mask(nodules , mask))
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
    }

    LOG_OUT(  "done.\n");
    return 0;
}