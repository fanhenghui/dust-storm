#pragma once


#include <string>

#define TUMOR_TYPE_NUM 22
const std::string TUMOR_TYPES[TUMOR_TYPE_NUM] =
{
    "LGIEN", //低级别上皮内瘤变 腺瘤
    "HGIEN", //高级别上皮内瘤变 腺瘤
    "PRE_LGIEN",//低级别上皮内瘤变 上皮内瘤变
    "PRE_HGIEN",//高级别上皮内瘤变 上皮内瘤变
    "papillary_adenoma", //乳头状腺癌
    "tubular_adenoma", //管状腺癌
    "mucinous_adenoma", //粘液腺癌
    "low_adhesion_adenoma", //低粘附性癌
    "mixed_adenoma", //混合性腺癌
    "adenosquamous_carcinoma", //腺鳞癌
    "medullary_carcinoma", //伴有淋巴样间质的癌（髓样癌）
    "hepatoid_adenocarcinoma", //肝样腺癌
    "squamous-cell_carcinoma", //鳞状细胞癌
    "undifferentiated_carcinoma", //未分化癌
    "NET1", //NET1级（类癌）
    "NET2", //NET2级
    "NEC_big_cell", //大细胞NEC
    "NEC_small_cell", //小细胞NEC
    "MIX_NET" ,//混合性腺神经内分泌癌
    "NO_GEIEN",//无上皮内瘤变
    "Uncertain_GEIEN",//不确定的上皮内瘤变
    "Uncertain",//不确定的病变
};

struct AnnotationFileHeader
{
    int group_num;
    int anno_num;
    int valid;

    AnnotationFileHeader()
    {
        group_num = 0;
        anno_num = 0;
        valid = 0;
    }
};

struct GroupUnit
{
    char name_str[256];
    char group_name_str[256]; //all 0 has no group
    char color_str[128];

    GroupUnit()
    {
        memset(name_str, 0, sizeof(name_str));
        memset(group_name_str, 0, sizeof(group_name_str));
        memset(color_str, 0, sizeof(color_str));
    }
};

struct AnnotationUnit
{
    char name_str[256];
    char group_name_str[256];
    char color_str[128];

    int anno_type_id;
    unsigned char entrypt_tumor_type_id[512];//unsigned char tumor_type_id[17]->entrypt_tumor_type_id[512]// no need to encryption
    unsigned int point_num;

    AnnotationUnit()
    {
        memset(name_str, 0, sizeof(name_str));
        memset(group_name_str, 0, sizeof(group_name_str));
        memset(color_str, 0, sizeof(color_str));
        anno_type_id = 0;
        memset(entrypt_tumor_type_id, 0, sizeof(entrypt_tumor_type_id));
        point_num = 0;
    }
};

inline int get_tumor_type_id(const std::string& tumor_type)
{
    for (int i = 0; i < TUMOR_TYPE_NUM; ++i)
    {
        if (tumor_type == TUMOR_TYPES[i])
        {
            return i;
        }
    }
    return -1;
}

inline void char_array_to_string(char* char_array, int length, std::string& s)
{
    s.clear();
    for (int i = 0; i < length; ++i)
    {
        if (char_array[i] == '\0')
        {
            break;
        }
        s.push_back(char_array[i]);
    }
}