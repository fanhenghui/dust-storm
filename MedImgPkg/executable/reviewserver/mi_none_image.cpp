#include "mi_none_image.h"
#include "mi_message.pb.h"
#include "mi_review_logger.h"

MED_IMG_BEGIN_NAMESPACE

void NoneImgCircles::add_circle(const NoneImgCircles::CircleUnit& circle) {
    _circles.push_back(circle);
}

void NoneImgCircles::set_circle(const std::vector<CircleUnit> circles) {
    _circles = circles;
}
const std::vector<NoneImgCircles::CircleUnit>& NoneImgCircles::get_circles() const {
    return _circles;
}

char* NoneImgCircles::serialize_to_array(int &bytelength) {
    MsgNoneImgCircles msg;
    for (size_t i = 0; i< _circles.size(); ++i) {
        MsgCircleUnit* unit = msg.add_circles();
        unit->set_cx(_circles[i].cx);
        unit->set_cy(_circles[i].cy);
        unit->set_r(_circles[i].r);
    }

    bytelength = msg.ByteSize();
    if(bytelength == 0) {
        MI_REVIEW_LOG(MI_ERROR) << "serialize none-img-circles: byte length is 0.";
        return nullptr;
    }
    char* data = new char[bytelength];
    if (!msg.SerializeToArray(data, bytelength)) {
        MI_REVIEW_LOG(MI_ERROR) << "serialize none-img-circles: serialize failed.";
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
        MI_REVIEW_LOG(MI_ERROR) << "serialize none-img-cornerinfo: byte length is 0.";
        return nullptr;
    }
    char* data = new char[bytelength];
    if (!msg.SerializeToArray(data, bytelength)) {
        MI_REVIEW_LOG(MI_ERROR) << "serialize none-img-cornerinfo: serialize failed.";
        delete [] data;
        bytelength = 0;
        return nullptr; 
    } else {
        return data;
    }
}

void NoneImgCollection::set_circles(std::shared_ptr<NoneImgCircles> circles) {
    _circles = circles;
}

void NoneImgCollection::set_corner_infos(std::shared_ptr<NoneImgCornerInfos> corner_infos) {
    _corner_infos = corner_infos;
}

char* NoneImgCollection::serialize_to_array(int &bytelength) {
    MsgNoneImgCollection msg;
    if(_circles == nullptr && _corner_infos == nullptr) {
        return nullptr;
    }

    if(_circles) {
        MsgNoneImgCircles* circles_msg = msg.mutable_circles();
        auto circles = _circles->get_circles();
        for (size_t i = 0; i< circles.size(); ++i) {
            MsgCircleUnit* unit = circles_msg->add_circles();
            unit->set_cx(circles[i].cx);
            unit->set_cy(circles[i].cy);
            unit->set_r(circles[i].r);
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
        MI_REVIEW_LOG(MI_ERROR) << "serialize none-img-collection: byte length is 0.";
        return nullptr;
    }
    char* data = new char[bytelength];
    if (!msg.SerializeToArray(data, bytelength)) {
        MI_REVIEW_LOG(MI_ERROR) << "serialize none-img-collection: serialize failed.";
        delete [] data;
        bytelength = 0;
        return nullptr; 
    } else {
        return data;
    }
}

MED_IMG_END_NAMESPACE