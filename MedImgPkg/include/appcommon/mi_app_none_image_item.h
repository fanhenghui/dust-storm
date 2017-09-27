#ifndef MED_IMG_APPCOMMON_MI_APP_NONE_IMAGE_ITEM_H
#define MED_IMG_APPCOMMON_MI_APP_NONE_IMAGE_ITEM_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "appcommon/mi_app_common_export.h"

MED_IMG_BEGIN_NAMESPACE

class INoneImg {
public:
    INoneImg(const std::string& type):_type(type) {};
    virtual ~INoneImg() {};
    virtual char* serialize_to_array(int &bytelength) = 0;

    std::string get_type() const {
        return _type;
    };

    void set_name(const std::string& name) {
        _name = name;
    };/*  */
    std::string get_name() const {
        return _name;
    };

private:
    std::string _type;
    std::string _name;
};

class NoneImgCircles : public INoneImg {
public:
    struct CircleUnit {
        float cx;//center x
        float cy;//center y
        float r;//radius
    };

public:
    NoneImgCircles():INoneImg("circles") {};
    virtual ~NoneImgCircles() {};
    virtual char* serialize_to_array(int &bytelength);

    void add_circle(const CircleUnit& circle);
    void set_circle(const std::vector<CircleUnit> circles);
    const std::vector<CircleUnit>& get_circles() const;

private:
    std::vector<CircleUnit> _circles;
};

// LT|1:patientName|2:patientID\n
class NoneImgCornerInfos : public INoneImg {
public:
    enum PositionType {
        LT = 0,
        LB = 1,
        RT = 2,
        RB = 3,
    };

public:
    NoneImgCornerInfos():INoneImg("cornerinfos") {};
    virtual ~NoneImgCornerInfos() {};
    virtual char* serialize_to_array(int &bytelength);

    void set_infos(PositionType pos, std::vector<std::pair<int, std::string>> infos);
    void add_info(PositionType pos, std::pair<int, std::string> info);
    std::string to_string() const;

private:
    std::map<PositionType,std::vector<std::pair<int, std::string>>> _infos;
};

class NoneImgCollection: public INoneImg {
public:
    NoneImgCollection():INoneImg("cornerinfos") {};
    virtual ~NoneImgCollection() {};
    virtual char* serialize_to_array(int &bytelength);

    void set_circles(std::shared_ptr<NoneImgCircles> circles);
    void set_corner_infos(std::shared_ptr<NoneImgCornerInfos> corner_infos);
private:
    std::shared_ptr<NoneImgCircles> _circles;
    std::shared_ptr<NoneImgCornerInfos> _corner_infos;
};


MED_IMG_END_NAMESPACE


#endif