#include "mi_app_none_image_item.h"
#include "mi_app_common_logger.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

void NoneImgAnnotations::add_annotation(const NoneImgAnnotations::AnnotationUnit& anno) {
    _annotations.push_back(anno);
}

void NoneImgAnnotations::set_annotations(const std::vector<NoneImgAnnotations::AnnotationUnit> annotations) {
    _annotations = annotations;
}
const std::vector<NoneImgAnnotations::AnnotationUnit>& NoneImgAnnotations::get_annotations() const {
    return _annotations;
}

char* NoneImgAnnotations::serialize_to_array(int &bytelength) {
    MsgNoneImgAnnotations msg;
    for (size_t i = 0; i< _annotations.size(); ++i) {
        MsgAnnotationUnit* unit = msg.add_annotation();
        unit->set_type(_annotations[i].type);
        unit->set_id(_annotations[i].id);
        unit->set_status(_annotations[i].status);
        unit->set_visibility(_annotations[i].visibility);
        unit->set_para0(_annotations[i].para0);
        unit->set_para1(_annotations[i].para1);
        unit->set_para2(_annotations[i].para2);
    }

    bytelength = msg.ByteSize();
    if(bytelength == 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-annotations: byte length is 0.";
        return nullptr;
    }
    char* data = new char[bytelength];
    if (!msg.SerializeToArray(data, bytelength)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-annotations: serialize failed.";
        delete [] data;
        bytelength = 0;
        return nullptr; 
    } else {
        return data;
    }
}

void NoneImgCornerInfos::set_infos(NoneImgCornerInfos::PositionType pos, std::vector<std::pair<int, std::string>> infos) {
    _infos[pos] = infos;
}

void NoneImgCornerInfos::add_info(NoneImgCornerInfos::PositionType pos, std::pair<int, std::string> info) {
    auto it = _infos.find(pos);
    if (it == _infos.end()) {
        _infos[pos] = std::vector<std::pair<int, std::string>>(1,info);
    } else {
        _infos[pos].push_back(info);
    }
}

std::string NoneImgCornerInfos::to_string() const {
    //struct infos to character array
    //Format "LT|1:patientName|2:patientID\nLB....\nRT|....\nRB|....\n"
    typedef std::map<NoneImgCornerInfos::PositionType,std::vector<std::pair<int, std::string>>>::const_iterator const_it;
    std::stringstream ss;
    static const std::string pos_str[4] = {"LT", "LB", "RT", "RB"};
    for (const_it it = _infos.begin(); it != _infos.end(); ++it) {
        NoneImgCornerInfos::PositionType pos = it->first;
        if(!it->second.empty()) {
            ss << pos_str[static_cast<int>(pos)] << "|";
            for (size_t i = 0; i < it->second.size(); ++i) {
                ss << "|" << it->second[i].first << ":" << it->second[i].second;
            }
            ss << "\n";
            MI_APPCOMMON_LOG(MI_DEBUG) <<"corner info str: " << ss.str();
        }
    }

    return ss.str();
}

char* NoneImgCornerInfos::serialize_to_array(int &bytelength) {
    const std::string str = this->to_string();
    if(str.empty()) {
        return nullptr;
    }

    //serialize
    MsgNoneImgCornerInfos msg;
    msg.set_infos(str);

    bytelength = msg.ByteSize();
    if(bytelength == 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-cornerinfo: byte length is 0.";
        return nullptr;
    }
    char* data = new char[bytelength];
    if (!msg.SerializeToArray(data, bytelength)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-cornerinfo: serialize failed.";
        delete [] data;
        bytelength = 0;
        return nullptr; 
    } else {
        return data;
    }
}

void NoneImgCollection::set_annotations(std::shared_ptr<NoneImgAnnotations> annotations) {
    _annotations = annotations;
}

void NoneImgCollection::set_corner_infos(std::shared_ptr<NoneImgCornerInfos> corner_infos) {
    _corner_infos = corner_infos;
}

char* NoneImgCollection::serialize_to_array(int &bytelength) {
    MsgNoneImgCollection msg;
    if(_annotations == nullptr && _corner_infos == nullptr) {
        return nullptr;
    }

    if(_annotations) {
        MsgNoneImgAnnotations* annotations_msg = msg.mutable_annotations();
        auto annotations = _annotations->get_annotations();
        for (size_t i = 0; i< annotations.size(); ++i) {
            MsgAnnotationUnit* unit = annotations_msg->add_annotation();
            unit->set_type(annotations[i].type);
            unit->set_id(annotations[i].id);
            unit->set_status(annotations[i].status);
            unit->set_visibility(annotations[i].visibility);
            unit->set_para0(annotations[i].para0);
            unit->set_para1(annotations[i].para1);
            unit->set_para2(annotations[i].para2);
        }
    }
    if(_corner_infos) { 
        MsgNoneImgCornerInfos* corner_msg = msg.mutable_corner_infos();
        const std::string str = _corner_infos->to_string();
        if(!str.empty()) {
            corner_msg->set_infos(str);
        }
    }

    bytelength = msg.ByteSize();
    if(bytelength == 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-collection: byte length is 0.";
        return nullptr;
    }
    char* data = new char[bytelength];
    if (!msg.SerializeToArray(data, bytelength)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-collection: serialize failed.";
        delete [] data;
        bytelength = 0;
        return nullptr; 
    } else {
        return data;
    }
}

MED_IMG_END_NAMESPACE