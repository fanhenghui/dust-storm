#pragma once


#include <string>

#define TUMOR_TYPE_NUM 22
const std::string TUMOR_TYPES[TUMOR_TYPE_NUM] =
{
    "LGIEN", //�ͼ�����Ƥ������ ����
    "HGIEN", //�߼�����Ƥ������ ����
    "PRE_LGIEN",//�ͼ�����Ƥ������ ��Ƥ������
    "PRE_HGIEN",//�߼�����Ƥ������ ��Ƥ������
    "papillary_adenoma", //��ͷ״�ٰ�
    "tubular_adenoma", //��״�ٰ�
    "mucinous_adenoma", //ճҺ�ٰ�
    "low_adhesion_adenoma", //��ճ���԰�
    "mixed_adenoma", //������ٰ�
    "adenosquamous_carcinoma", //���۰�
    "medullary_carcinoma", //�����ܰ������ʵİ�����������
    "hepatoid_adenocarcinoma", //�����ٰ�
    "squamous-cell_carcinoma", //��״ϸ����
    "undifferentiated_carcinoma", //δ�ֻ���
    "NET1", //NET1�����఩��
    "NET2", //NET2��
    "NEC_big_cell", //��ϸ��NEC
    "NEC_small_cell", //Сϸ��NEC
    "MIX_NET" ,//����������ڷ��ڰ�
    "NO_GEIEN",//����Ƥ������
    "Uncertain_GEIEN",//��ȷ������Ƥ������
    "Uncertain",//��ȷ���Ĳ���
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