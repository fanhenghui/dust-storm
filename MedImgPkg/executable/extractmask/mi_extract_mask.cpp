#include <string>
#include <fstream>
#include <vector>

#include "util/mi_file_util.h"
#include "util/mi_string_number_converter.h"
#include "arithmetic/mi_scan_line_analysis.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_intersection_test.h"
#include "io/mi_image_data_header.h"
#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_run_length_operator.h"
#include "io/mi_jpeg_parser.h"
#include "pugixml/pugixml.hpp"
#include "pugixml/pugiconfig.hpp"

#include "mi_extract_mask_common.h"

using namespace medical_imaging;

//////////////////////////////////////////////////////////////////////////
static std::ofstream out_log;
class LogSheild {
public:
    LogSheild(const std::string& log_file , const std::string& start_word) {
        out_log.open("anon.log" , std::ios::out);

        if (out_log.is_open()) {
            out_log << start_word;
        }
    }
    ~LogSheild() {
        out_log.close();
    }
protected:
private:
};


#define LOG_OUT(info) std::cout << info; out_log << info;
//////////////////////////////////////////////////////////////////////////


int save_mask(std::shared_ptr<ImageData> mask , const std::string& path , bool compressed) {
    if (path.empty()) {
        LOG_OUT("save mask path is empty!");
        return -1;
    }

    if (compressed) {
        std::vector<unsigned int> code = RunLengthOperator::encode((unsigned char*)mask->get_pixel_pointer()
                                         , mask->get_data_size());

        if (code.empty()) {
            LOG_OUT("mask is all zero!");
            return -1;
        }

        return FileUtil::write_raw(path , code.data() ,
                                   static_cast<unsigned int>(code.size()) * sizeof(unsigned int));
    } else {
        return FileUtil::write_raw(path , mask->get_pixel_pointer() , mask->get_data_size());
    }
}

int load_dicom_series(std::vector<std::string>& files ,
                      std::shared_ptr<ImageDataHeader>& header,
                      std::shared_ptr<ImageData>& img) {
    DICOMLoader loader;

    if (loader.load_series(files , img , header) == IO_SUCCESS) {
        return 0;
    } else {
        return -1;
    }
}

int get_nodule_set(const std::string& xml_file, std::vector<std::vector<Nodule>>& nodules ,
                   std::string& series_uid) {
    pugi::xml_document doc;

    if (!doc.load_file(xml_file.c_str())) {
        LOG_OUT("Load file failed!\n!");
        return -1;
    }

    //Root
    pugi::xml_node root_node = doc.child("LidcReadMessage");

    if (root_node.empty()) {
        LOG_OUT("invalid format , find Lidc read message failed!\n");
        return -1;
    }

    //Response header
    pugi::xml_node response_header_node = root_node.child("ResponseHeader");

    if (response_header_node.empty()) {
        LOG_OUT("invalid format , find Lidc response header failed!\n");
        return -1;
    }

    //Study UID
    pugi::xml_node study_uid_node = response_header_node.child("StudyInstanceUID");

    if (study_uid_node.empty()) {
        LOG_OUT("invalid format , find study UID failed!\n");
        return -1;
    }

    const std::string study_uid = study_uid_node.child_value();
    //LOG_OUT( std::string("study uid : ") + study_uid + "\n");

    //Series UID
    pugi::xml_node series_uid_node = response_header_node.child("SeriesInstanceUid");

    if (series_uid_node.empty()) {
        LOG_OUT("invalid format , find series UID failed!\n");
        return -1;
    }

    series_uid = series_uid_node.child_value();

    //LOG_OUT( "series uid : " + series_uid + "\n");
    if (series_uid.empty()) {
        LOG_OUT("invalid format , empty series UID !\n");
        return -1;
    }



    StrNumConverter<double> str_to_num;
    pugi::xpath_node_set reading_session_node_set =
        root_node.select_nodes("readingSession");// a reading session means a certain reader's result

    for (auto it = reading_session_node_set.begin() ; it != reading_session_node_set.end(); ++it) {
        std::vector<Nodule> nodules_perreader;
        pugi::xpath_node_set unblind_nodule_node_set = (*it).node().select_nodes("unblindedReadNodule");

        for (auto it2 = unblind_nodule_node_set.begin() ; it2 != unblind_nodule_node_set.end(); ++it2) {
            pugi::xpath_node node_unblinded = (*it2);
            Nodule nodule;
            nodule.type = 0;

            //get ID
            pugi::xml_node id_node = node_unblinded.node().child("noduleID");

            if (id_node.empty()) {
                //TODO ERROR
                LOG_OUT("invalid format , find nodule ID failed!\n");
                return -1;
            }

            nodule.name = id_node.child_value();

            //get characteristics TODO

            //get contour
            pugi::xpath_node_set roi_node_set = node_unblinded.node().select_nodes("roi");

            for (auto it3 = roi_node_set.begin() ; it3 != roi_node_set.end(); ++it3) {
                pugi::xpath_node roi_node = (*it3);
                pugi::xml_node pos_z_node = roi_node.node().child("imageZposition");

                if (pos_z_node.empty()) {
                    //TODO ERROR
                    LOG_OUT("invalid format , find image position z failed!\n");
                    return -1;
                }

                const double pos_z = str_to_num.to_num(pos_z_node.child_value());

                pugi::xpath_node_set edge_node_set = roi_node.node().select_nodes("edgeMap");

                for (auto it4 = edge_node_set.begin() ; it4 != edge_node_set.end(); ++it4) {
                    pugi::xpath_node edge_node = (*it4);
                    pugi::xml_node x_node = edge_node.node().child("xCoord");

                    if (x_node.empty()) {
                        //TODO ERROR
                        LOG_OUT("invalid format , find edge position x failed!\n");
                        return -1;
                    }

                    pugi::xml_node y_node = edge_node.node().child("yCoord");

                    if (y_node.empty()) {
                        //TODO ERROR
                        LOG_OUT("invalid format , find edge position y failed!\n");
                        return -1;
                    }

                    const double pos_x = str_to_num.to_num(x_node.child_value());
                    const double pos_y = str_to_num.to_num(y_node.child_value());
                    nodule._points.push_back(Point3(pos_x , pos_y , pos_z));
                }
            }

            nodules_perreader.push_back(nodule);
        }

        pugi::xpath_node_set non_nodule_node_set = (*it).node().select_nodes("nonNodule");

        for (auto it2 = non_nodule_node_set.begin() ; it2 != non_nodule_node_set.end(); ++it2) {
            pugi::xpath_node node_nonnodule = (*it2);
            Nodule nodule;
            nodule.type = 1;//for non-noudle

            //get ID
            pugi::xml_node id_node = node_nonnodule.node().child("nonNoduleID");

            if (id_node.empty()) {
                //TODO ERROR
                LOG_OUT("invalid format , find non-nodule ID failed!\n");
                return -1;
            }

            nodule.name = id_node.child_value();

            //get position z
            pugi::xml_node pos_z_node = node_nonnodule.node().child("imageZposition");

            if (pos_z_node.empty()) {
                //TODO ERROR
                LOG_OUT("invalid format , find image position z failed!\n");
                return -1;
            }

            const double pos_z = str_to_num.to_num(pos_z_node.child_value());

            pugi::xml_node locus_node = node_nonnodule.node().child("locus");

            if (locus_node.empty()) {
                //TODO ERROR
                LOG_OUT("invalid format , find non-nodule locus failed!\n");
                return -1;
            }

            pugi::xml_node x_node = locus_node.child("xCoord");
            pugi::xml_node y_node = locus_node.child("yCoord");


            const double pos_x = str_to_num.to_num(x_node.child_value());
            const double pos_y = str_to_num.to_num(y_node.child_value());
            nodule._points.push_back(Point3(pos_x , pos_y , pos_z));

            nodules_perreader.push_back(nodule);
        }

        nodules.push_back(nodules_perreader);
    }

    return 0;
}

#define LOCATION_EPSILON 0.0001
//if slice location is not match with position z , use position z (not very precise)
int resample_z_backup(std::vector<std::vector<Nodule>>& nodules ,
                      std::shared_ptr<ImageDataHeader>& header , bool save_slice_location_less) {
    std::vector<double> slice_location;

    for (int i = 0 ; i < header->image_position.size() ; ++i) {
        slice_location.push_back(header->image_position[i].z);
    }

    if (save_slice_location_less) {
        std::sort(slice_location.begin() , slice_location.end() ,
                  std::less<double>());//slice location from max to min
    } else {
        std::sort(slice_location.begin() , slice_location.end() ,
                  std::greater<double>());//slice location from max to min
    }


    double delta = 0;

    for (int i = 1; i < slice_location.size() ; ++i) {
        if (fabs(slice_location[i] - slice_location[i - 1]) > LOCATION_EPSILON) {
            delta = slice_location[i] - slice_location[i - 1];
        }
    }

    const double slice0 = slice_location[0];


    for (auto itreader = nodules.begin() ; itreader != nodules.end() ; ++itreader) {
        for (auto it = (*itreader).begin()  ; it != (*itreader).end() ; ++it) {
            Nodule& nodule = *it;
            std::vector<Point3>& pts = nodule._points;


            for (int  i = 0 ; i < pts.size() ; ++i) {
                const double slice = pts[i].z;

                double min_error = std::numeric_limits<double>::max();
                int min_id = 0;

                for (int j = 0 ; j < slice_location.size() ; ++j) {
                    double err = fabs(slice_location[j] - slice);

                    if (err < min_error) {
                        min_error = err;
                        min_id = j;
                    }

                    if (min_error < LOCATION_EPSILON) {
                        pts[i].z = static_cast<double>(min_id);
                        goto FIND_LOCATION;
                    }
                }

                pts[i].z = static_cast<double>(min_id);

                LOG_OUT("warning : find slice location gap more than 0.0001 !\n");
                //return -1;

FIND_LOCATION:
                ;
            }

        }

    }

    return 0;

}

int resample_z(std::vector<std::vector<Nodule>>& nodules ,
               std::shared_ptr<ImageDataHeader>& header , bool save_slice_location_less) {
    std::vector<double> slice_location = header->slice_location;

    if (!save_slice_location_less) {
        std::sort(slice_location.begin() , slice_location.end() ,
                  std::greater<double>());//slice location from max to min
    }

    double delta = 0;

    for (int i = 1; i < slice_location.size() ; ++i) {
        if (fabs(slice_location[i] - slice_location[i - 1]) > LOCATION_EPSILON) {
            delta = slice_location[i] - slice_location[i - 1];
        }
    }

    if (fabs(delta) < LOCATION_EPSILON) {
        LOG_OUT("slice location error!\n");
        return -1;
    }

    const double slice0 = slice_location[0];

    for (auto itreader = nodules.begin() ; itreader != nodules.end() ; ++itreader) {
        for (auto it = (*itreader).begin()  ; it != (*itreader).end() ; ++it) {
            Nodule& nodule = *it;
            std::vector<Point3>& pts = nodule._points;

            for (int  i = 0 ; i < pts.size() ; ++i) {
                double slice = pts[i].z;
                double delta_slice = slice - slice0;
                int tmp_idx = static_cast<int>(delta_slice / delta);

                if (tmp_idx > slice_location.size()) {
                    LOG_OUT("find slice lotation failed!\n");
                    return -1;
                }

                if (fabs(slice_location[tmp_idx] - slice) < LOCATION_EPSILON) {
                    pts[i].z = static_cast<double>(tmp_idx);
                    goto FIND_LOCATION;
                } else if (slice_location[tmp_idx] - slice < 0) {
                    if (save_slice_location_less) {
                        for (int j = tmp_idx ; j < slice_location.size() ; ++j) {
                            if (fabs(slice_location[j] - slice) < LOCATION_EPSILON) {
                                pts[i].z = static_cast<double>(j);
                                goto FIND_LOCATION;
                            }
                        }
                    } else {
                        for (int j = tmp_idx ; j > 0 ; --j) {
                            if (fabs(slice_location[j] - slice) < LOCATION_EPSILON) {
                                pts[i].z = static_cast<double>(j);
                                goto FIND_LOCATION;
                            }
                        }
                    }

                    LOG_OUT("find slice lotation failed!\n");
                    return -1;
                } else {
                    if (save_slice_location_less) {
                        for (int j = tmp_idx ; j > 0 ; --j) {
                            if (fabs(slice_location[j] - slice) < LOCATION_EPSILON) {
                                pts[i].z = static_cast<double>(j);
                                goto FIND_LOCATION;
                            }
                        }
                    } else {
                        for (int j = tmp_idx ; j < slice_location.size() ; ++j) {
                            if (fabs(slice_location[j] - slice) < LOCATION_EPSILON) {
                                pts[i].z = static_cast<double>(j);
                                goto FIND_LOCATION;
                            }
                        }
                    }


                    LOG_OUT("find slice lotation failed!\n");
                    return -1;
                }

FIND_LOCATION:
                ;
            }
        }

    }

    return 0;
}

void cal_nodule_aabb(std::vector <std::vector<Nodule>>& nodules) {
    for (auto itreader = nodules.begin() ; itreader != nodules.end() ; ++itreader) {
        for (auto it = (*itreader).begin()  ; it != (*itreader).end() ; ++it) {
            Nodule& nodule = *it;
            const std::vector<Point3>& pts = nodule._points;

            nodule._aabb._min[0] = static_cast<int>(pts[0].x);
            nodule._aabb._min[1] = static_cast<int>(pts[0].y);
            nodule._aabb._min[2] = static_cast<int>(pts[0].z);

            nodule._aabb._max[0]  = nodule._aabb._min[0];
            nodule._aabb._max[1]  = nodule._aabb._min[1];
            nodule._aabb._max[2]  = nodule._aabb._min[2];

            for (int i = 1 ; i < pts.size() ; ++i) {
                int tmp[3] = { static_cast< int>(pts[i].x) , static_cast<int>(pts[i].y), static_cast<int>(pts[i].z)};

                for (int  j = 0 ; j < 3 ; ++j) {
                    nodule._aabb._min[j] = nodule._aabb._min[j] > tmp[j] ? tmp[j] : nodule._aabb._min[j];
                    nodule._aabb._max[j] = nodule._aabb._max[j] < tmp[j] ? tmp[j] : nodule._aabb._max[j];
                }
            }
        }
    }
}

void get_same_region_nodules(std::vector <std::vector<Nodule>>::iterator reader , Nodule& target ,
                             std::vector <std::vector<Nodule>>& nodules, std::vector<Nodule*>& same_nodules,
                             std::vector<int>& reader_id ,  float region_percent) {
    same_nodules.clear();
    reader_id.clear();

    const int v_target = target._aabb.volume();

    int cur_readid = 0;


    for (auto it = nodules.begin() ; it != nodules.end(); ++it , ++cur_readid) {
        if (it == reader) { //skip myself
            continue;
        }

        Nodule* cross_nodule = nullptr;
        float max_region_percent = -1;

        for (auto it2 = (*it).begin()  ; it2 != (*it).end() ; ++it2) {
            Nodule& nodule = *it2;

            if (nodule.type != 0 || nodule._points.size() < 2) {
                continue;
            }

            if (nodule.flag != 0) { //has been analysized
                continue;
            }

            AABBI inter;

            if (IntersectionTest::aabb_to_aabb(nodule._aabb , target._aabb , inter)) {
                const int v = nodule._aabb.volume();
                const int inter_v = inter.volume();
                const float inter_p = static_cast<float>(inter_v) / ((v + v_target) * 0.5f);

                //std::cout << "cross percent : " << inter_p << std::endl;
                if (inter_p > max_region_percent) {
                    max_region_percent = inter_p;
                    cross_nodule = &nodule;
                }
            }
        }

        if (max_region_percent > region_percent) {
            same_nodules.push_back(cross_nodule);
            reader_id.push_back(cur_readid);
        }

    }

}

typedef ScanLineAnalysis<unsigned char>::Pt2 PT2;
void scan_contour_to_mask(std::vector<Point3>& pts , std::shared_ptr<ImageData> mask,
                          unsigned char label) {
    ScanLineAnalysis<unsigned char> scan_line_analysis;

    bool begin = true;
    int current_z = 0;
    std::vector<PT2> current_contour;

    for (int i = 0 ; i < pts.size() ; ++i) {
        if (begin) {
            current_z = static_cast<int>(pts[i].z);
            current_contour.push_back(PT2(static_cast<int>(pts[i].x) , static_cast<int>(pts[i].y)));
            begin = false;
        } else {
            int z = static_cast<int>(pts[i].z);

            if (z == current_z) { //push contour
                current_contour.push_back(PT2(static_cast<int>(pts[i].x) , static_cast<int>(pts[i].y)));
            } else { // do scaning
                scan_line_analysis.fill((unsigned char*)mask->get_pixel_pointer() + current_z *
                                        mask->_dim[0]*mask->_dim[1] ,
                                        mask->_dim[0] , mask->_dim[1] , current_contour , label);

                //back to begin
                --i;
                current_contour.clear();
                begin = true;
            }
        }
    }
}

int contour_to_mask(std::vector <std::vector<Nodule>>& nodules , std::shared_ptr<ImageData> mask,
                    float same_nodule_precent , int confidence, int pixel_confidence , int setlogic) {
    mask->_data_type = UCHAR;
    mask->mem_allocate();

    std::vector<std::shared_ptr<ImageData>> mask_reader;
    mask_reader.resize(nodules.size());
    mask_reader[0] = mask;

    for (int i = 1; i < nodules.size() ; ++i) {
        mask_reader[i] = std::shared_ptr<ImageData>(new ImageData);
        mask->shallow_copy(mask_reader[i].get());
        mask_reader[i]->mem_allocate();
    }

    ScanLineAnalysis<unsigned char> scan_line_analysis;
    unsigned char label = 0;

    for (auto itreader = nodules.begin() ; itreader != nodules.end() ; ++itreader) {
        for (auto it = (*itreader).begin()  ; it != (*itreader).end() ; ++it) {
            Nodule& nodule = *it;
            std::vector<Point3>& pts = nodule._points;

            if (pts.size() <= 1) { //TODO skip nodule < 3mm and non-nodule
                continue;
            }

            ++label;

            if (nodule.flag != 0) { //has already been analysized
                continue;
            }

            //search the same region nodules
            std::vector<Nodule*> same_nodules;
            std::vector<int> reader_id;
            get_same_region_nodules(itreader , nodule , nodules , same_nodules , reader_id ,
                                    same_nodule_precent);

            //case 1 just one reader annotation this nodule > 3mm
            int cur_confidence = 1 + static_cast<int>(same_nodules.size());

            if (cur_confidence < confidence) {
                nodule.flag = -1;//not satisify confidence

                for (int k = 0 ; k < same_nodules.size() ; ++k) {
                    same_nodules[k]->flag = -1;
                }

                continue;
            }

            //set flags
            nodule.flag = label;

            for (int k = 0 ; k < same_nodules.size() ; ++k) {
                same_nodules[k]->flag = label;
            }


            //choose points based on set logic(intersection or union)
            if (setlogic == 0) { //intersection
                scan_contour_to_mask(pts , mask, label);

                //save_mask( mask , "D:/temp/0.raw" ,false);

                for (int k = 0 ; k < same_nodules.size() ; ++k) {
                    std::vector<Point3>& pts_same_nodule = same_nodules[k]->_points;
                    scan_contour_to_mask(pts_same_nodule , mask_reader[reader_id[k]], label);

                    //StrNumConverter<int> conv;
                    //save_mask( mask_reader[reader_id[k]] , std::string("D:/temp/") + conv.to_string(reader_id[k]) + ".raw" ,false);
                }

                //extracted label from label to interlabel(label + confidence)
                unsigned char inter_label = label + cur_confidence;
                AABBI max_region = nodule._aabb;

                for (int k = 0 ; k < same_nodules.size() ; ++k) {
                    AABBI sub_regio = same_nodules[k]->_aabb;

                    for (int k2 = 0  ; k2 < 3 ; ++k2) {
                        max_region._min[k2] = max_region._min[k2] > sub_regio._min[k2] ?
                                              sub_regio._min[k2] : max_region._min[k2];

                        max_region._max[k2] = max_region._max[k2] < sub_regio._max[k2] ?
                                              sub_regio._max[k2] : max_region._max[k2];
                    }
                }

                //reset uninterceted position to 0 , and set interceted position to label
                unsigned char* mask_data = (unsigned char*)mask->get_pixel_pointer();

                for (int z = max_region._min[2] ; z <= max_region._max[2] ; ++z) {
                    for (int y = max_region._min[1] ; y <= max_region._max[1] ; ++y) {
                        for (int x = max_region._min[0] ; x <= max_region._max[0] ; ++x) {
                            const int idx = z * mask->_dim[0] * mask->_dim[1] + y * mask->_dim[0] + x;

                            if (mask_data[idx] == label) {
                                int cur_inter = 1;

                                for (int k = 0 ; k < reader_id.size() ; ++k) {
                                    unsigned char* mask_other = (unsigned char*)mask_reader[k]->get_pixel_pointer();

                                    if (mask_other[idx] == label) {
                                        ++cur_inter;
                                    }
                                }

                                if (cur_inter < pixel_confidence) {
                                    mask_data[idx] = 0;
                                }

                            }
                        }
                    }
                }


            } else if (setlogic == 1) { //union
                scan_contour_to_mask(pts , mask, label);

                for (int k = 0 ; k < same_nodules.size() ; ++k) {
                    std::vector<Point3>& pts_same_nodule = same_nodules[k]->_points;
                    scan_contour_to_mask(pts_same_nodule , mask, label);
                }
            } else {
                LOG_OUT("Invalid nodule region set logic.");
                return -1;
            }


        }

    }


    return 0;

}


int browse_root_xml(const std::string& root , std::vector<std::string>& xml_files) {
    if (root.empty()) {
        LOG_OUT("empty root!");
        return -1;
    }

    std::map<std::string , std::vector<std::string> >files;
    std::set<std::string > postfix;
    postfix.insert(".xml");

    FileUtil::get_all_file_recursion(root , postfix , files);

    if (files.empty()) {
        LOG_OUT("empty files!")
        return -1;
    }

    if (files.find(".xml") != files.end()) {
        xml_files = files[".xml"];
    }

    return 0;
}

int browse_root_dcm(const std::string& root,
                    std::map<std::string, std::vector<std::string>>& dcm_files) {
    if (root.empty()) {
        LOG_OUT("empty root!");
        return -1;
    }

    std::map<std::string , std::vector<std::string> >files;
    std::set<std::string > postfix;
    postfix.insert(".dcm");

    FileUtil::get_all_file_recursion(root , postfix , files);

    if (files.empty()) {
        LOG_OUT("empty files!")
        return -1;
    }

    if (files.find(".dcm") != files.end()) {
        DICOMLoader loader;
        std::vector<std::string>& all_dcm_files = files.find(".dcm")->second;

        for (auto it = all_dcm_files.begin() ; it != all_dcm_files.end() ; ++it) {
            std::string study_uid;
            std::string series_uid;
            loader.check_series_uid(*it , study_uid , series_uid);

            if (dcm_files.find(series_uid) == dcm_files.end()) {
                std::vector<std::string> sub_dcm_files(1 , *it);
                dcm_files[series_uid] = sub_dcm_files;
            } else {
                dcm_files[series_uid].push_back(*it);
            }
        }
    }

    return 0;
}

void produce_one_test_jpeg(std::shared_ptr<ImageData> img , std::shared_ptr<ImageData> mask ,
                           int slice , std::string& save_path) {
    const float PRESET_CT_LUNGS_WW = 1500;
    const float PRESET_CT_LUNGS_WL = -400;

    float ww = PRESET_CT_LUNGS_WW;
    float wl = PRESET_CT_LUNGS_WL;

    const float min_wl = wl - img->_intercept - ww * 0.5f;

    const int w = img->_dim[0];
    const int h = img->_dim[1];
    std::unique_ptr<unsigned char[]> buffer(new unsigned char[w * h * 3]);

    if (img->_data_type == DataType::USHORT) {
        unsigned short* volume_data = (unsigned short*)img->get_pixel_pointer();
        unsigned char* mask_data = (unsigned char*)mask->get_pixel_pointer();

        for (int y  = 0 ; y < h; ++y) {
            int yy = h - y - 1;

            for (int x = 0; x < w ; ++x) {
                unsigned int idx = slice * w * h + yy * w + x;
                unsigned short v = volume_data[idx];
                float v0 = ((float)v  - min_wl) / ww;
                v0 = v0 > 1.0f ? 1.0f : v0;
                v0 = v0 < 0.0f ? 0.0f : v0;
                unsigned char rgb = static_cast<unsigned char>(v0 * 255.0f);

                if (mask_data[idx] != 0) {
                    float rrr = (float)rgb  + 125;
                    rrr = rrr > 255.0f ? 255.0f : rrr;
                    buffer.get()[(y * w + x) * 3] = static_cast<unsigned char>(rrr);
                    buffer.get()[(y * w + x) * 3 + 1] = rgb;
                    buffer.get()[(y * w + x) * 3 + 2] = rgb;
                } else {
                    buffer.get()[(y * w + x) * 3] = rgb;
                    buffer.get()[(y * w + x) * 3 + 1] = rgb;
                    buffer.get()[(y * w + x) * 3 + 2] = rgb;
                }
            }
        }
    } else if (img->_data_type == DataType::SHORT) {
        short* volume_data = (short*)img->get_pixel_pointer();
        unsigned char* mask_data = (unsigned char*)mask->get_pixel_pointer();

        for (int y  = 0 ; y < h ; ++y) {
            int yy = h - y - 1;

            for (int x = 0; x < w ; ++x) {
                unsigned int idx = slice * w * h + yy * w + x;
                short v = volume_data[idx];
                float v0 = ((float)v   - min_wl) / ww;
                v0 = v0 > 1.0f ? 1.0f : v0;
                v0 = v0 < 0.0f ? 0.0f : v0;
                unsigned char rgb = static_cast<unsigned char>(v0 * 255.0f);

                if (mask_data[idx] != 0) {
                    float rrr = (float)rgb  + 125;
                    rrr = rrr > 255.0f ? 255.0f : rrr;
                    buffer.get()[(y * w + x) * 3] = static_cast<unsigned char>(rrr);
                    buffer.get()[(y * w + x) * 3 + 1] = rgb;
                    buffer.get()[(y * w + x) * 3 + 2] = rgb;
                } else {
                    buffer.get()[(y * w + x) * 3] = rgb;
                    buffer.get()[(y * w + x) * 3 + 1] = rgb;
                    buffer.get()[(y * w + x) * 3 + 2] = rgb;
                }
            }
        }
    }

    //decode to jpeg
    JpegParser::write_to_jpeg(save_path , (char*)buffer.get() , img->_dim[0] , img->_dim[1]);
}

void produce_test_jpeg(std::shared_ptr<ImageData> img , std::shared_ptr<ImageData> mask ,
                       const std::string& base_name) {
    #pragma omp parallel for

    for (int i = 0; i < img->_dim[2] ; ++i) {
        unsigned char* raw_mask = (unsigned char*)mask->get_pixel_pointer();
        bool got_it = false;

        for (int j = 0; j < img->_dim[0]*img->_dim[1] ; ++j) {
            if (raw_mask[j + i * img->_dim[0]*img->_dim[1]] != 0) {
                got_it = true;
                break;
            }
        }

        if (got_it) {
            std::stringstream ss;
            ss << base_name << "_slice_" << i << ".jpeg";
            produce_one_test_jpeg(img , mask , i , ss.str());
        }
    }
}

int extract_mask(int argc , char* argv[] , std::shared_ptr<ImageData>& last_img ,
                 std::shared_ptr<ImageDataHeader>& last_header , std::vector<std::vector<Nodule>>& last_nodule) {
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

    LogSheild log_sheild("em.log" , "Extracting mask from LIDC data set >>> \n");

    std::string dcm_direction;
    std::string annotation_direction;
    std::string output_direction;
    bool compressed = false;
    bool save_slice_location_less = true;
    float cross_nodule_percent = 0.5f;
    int confidence = 2;
    int pixel_confidence = 2;
    int setlogic = 0;//0 for intersection 1 for union
    bool jpeg_output = false;

    if (argc == 1) {
        LOG_OUT("invalid arguments!\n");
        LOG_OUT("targuments list:\n");
        LOG_OUT("\t-data <path> : DICOM data root(.dcm)\n");
        LOG_OUT("\t-annotation <path> : annotation root(.xml)\n");
        LOG_OUT("\t-output <path] : save mask root\n");
        LOG_OUT("\t-compress : if mask is compressed\n");
        LOG_OUT("\t-slicelocation <less/greater> : default is less\n");
        LOG_OUT("\t-crosspercent<0.1~1> default 0.5\n");
        LOG_OUT("\t-confidence<1~4> default is 2\n");
        LOG_OUT("\t-pixelconfidence<1~4> default is 2\n");
        LOG_OUT("\t-setlogic<inter/union> default is inter\n");
        LOG_OUT("\t-vis: if vis contour in 2D\n");
        LOG_OUT("\t-jpegout : if vis mask&img blending slice");
        return -1;
    } else {
        for (int i = 1; i < argc ; ++i) {
            if (std::string(argv[i]) == "-help") {
                LOG_OUT("arguments list:\n");
                LOG_OUT("\t-data <path> : DICOM data root(.dcm)\n");
                LOG_OUT("\t-annotation <path> : annotation root(.xml)\n");
                LOG_OUT("\t-output <path] : save mask root\n");
                LOG_OUT("\t-compress : if mask is compressed\n");
                LOG_OUT("\t-slicelocation <less/greater> : default is less\n");
                LOG_OUT("\t-crosspercent<0.1~1> default 0.5\n");
                LOG_OUT("\t-confidence<1~4> default is 2\n");
                LOG_OUT("\t-pixelconfidence<1~4> default is 2\n");
                LOG_OUT("\t-setlogic<inter/union> default is inter\n");
                LOG_OUT("\t-vis: if vis contour in 2D\n");
                LOG_OUT("\t-jpegout : if vis mask&img blending slice");
                return 0;
            }

            if (std::string(argv[i]) == "-data") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                dcm_direction = std::string(argv[i + 1]);

                ++i;
            } else if (std::string(argv[i]) == "-annotation") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                annotation_direction = std::string(argv[i + 1]);
                ++i;
            } else if (std::string(argv[i]) == "-output") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                output_direction = std::string(argv[i + 1]);
                ++i;
            } else if (std::string(argv[i]) == "-compress") {
                compressed = true;
            } else if (std::string(argv[i]) == "-jpegout") {
                jpeg_output = true;
            } else if (std::string(argv[i]) == "-slicelocation") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                if (std::string(argv[i + 1]) == "less") {
                    save_slice_location_less = true;
                } else if (std::string(argv[i + 1]) == "greater") {
                    save_slice_location_less = false;
                } else {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                ++i;
            } else if (std::string(argv[i]) == "-crosspercent") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                StrNumConverter<float> conv;
                cross_nodule_percent = conv.to_num(std::string(argv[i + 1]));
                ++i;
            } else if (std::string(argv[i]) == "-confidence") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                StrNumConverter<int> conv;
                confidence = conv.to_num(std::string(argv[i + 1]));
                ++i;
            } else if (std::string(argv[i]) == "-pixelconfidence") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                StrNumConverter<int> conv;
                pixel_confidence = conv.to_num(std::string(argv[i + 1]));
                ++i;
            } else if (std::string(argv[i]) == "-setlogic") {
                if (i + 1 > argc - 1) {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                if (std::string(argv[i + 1]) == "inter") {
                    setlogic = 0;
                } else if (std::string(argv[i + 1]) == "union") {
                    setlogic = 1;
                } else {
                    LOG_OUT("invalid arguments!\n");
                    return -1;
                }

                ++i;
            }
        }
    }

    if (dcm_direction.empty() || annotation_direction.empty() || output_direction.empty()) {
        LOG_OUT("invalid empty direction!\n");
        return -1;
    }

    std::map<std::string , std::vector<std::string>> dcm_files;

    if (0 != browse_root_dcm(dcm_direction , dcm_files)) {
        LOG_OUT("browse dicom direction failed!\n");
        return -1;
    }

    std::vector<std::string> xml_files;

    if (0 != browse_root_xml(annotation_direction , xml_files)) {
        LOG_OUT("browse annotation direction failed!\n");
        return -1;
    }

    for (auto it = xml_files.begin() ; it != xml_files.end() ; ++it) {
        LOG_OUT("parse annotation file : " + *it + " >>>\n");

        //parse nodule annotation file
        std::vector<std::vector<Nodule>> nodules;
        std::string series_uid;

        if (0 !=  get_nodule_set(*it , nodules , series_uid)) {
            LOG_OUT("parse nodule annotation failed!\n");
            return -1;
        }

        LOG_OUT("series UID : " + series_uid + "\n");

        auto it_dcm = dcm_files.find(series_uid);

        if (it_dcm == dcm_files.end()) {
            LOG_OUT("can't find dcm files!\n");
            continue;
        }

        LOG_OUT("loading DICOM files to get image information :  >>>\n");

        //slice location to pixel coordinate
        std::shared_ptr<ImageDataHeader> data_header;
        std::shared_ptr<ImageData> volume_data;

        if (0 != load_dicom_series(it_dcm->second , data_header, volume_data)) {
            LOG_OUT("load series failed!\n");
            return -1;
        }

        //resample position z from slice location to pixel coordinate
        if (0 != resample_z(nodules , data_header , save_slice_location_less)) {
            LOG_OUT("warning : use image position z replace silce location.")

            if (0 != resample_z_backup(nodules , data_header , save_slice_location_less)) {
                LOG_OUT("resample z failed!\n");
                return -1;
            }
        }

        LOG_OUT("convert contour to mask >>>\n");

        //calculate aabb for extract mask
        cal_nodule_aabb(nodules);

        //contour to mask
        std::shared_ptr<ImageData> mask(new ImageData);
        volume_data->shallow_copy(mask.get());

        if (0 != contour_to_mask(nodules , mask , cross_nodule_percent , confidence, pixel_confidence ,
                                 setlogic)) {
            LOG_OUT("convert contour to mask failed!\n");
            return -1;
        }

        //save mask
        std::string output;

        if (compressed) {
            output = output_direction + "/" + series_uid + ".rle";
        } else {
            output = output_direction + "/" + series_uid + ".raw";
        }

        if (0 != save_mask(mask , output , compressed)) {
            LOG_OUT("save mask failed!\n");
            return -1;
        }

        if (jpeg_output) {
            produce_test_jpeg(volume_data , mask , output_direction + "/" + series_uid);
        }

        LOG_OUT("extract mask done.\n");

        last_img = volume_data;
        last_header = data_header;
        last_nodule = nodules;
    }

    LOG_OUT("done.\n");
    return 0;
}

int logic(int argc , char* argv[]) {
    std::shared_ptr<ImageData> last_img;
    std::shared_ptr<ImageDataHeader> last_header;
    std::vector<std::vector<Nodule>> last_nodule;
    return extract_mask(argc , argv , last_img , last_header , last_nodule);
}