#ifndef TUMOR_TYPE_RUIJIN_H
#define TUMOR_TYPE_RUIJIN_H

#include <string>

#define TUMOR_TYPE_NUM 24

//////////////////////////////////////////////////////////////////////////
//���listһ��Ҫ��UI�Ĵ�checkbox����˳��һ�£�Ҫ�����ַ���ֵ��Annotation_define.h�е�ֵ��ͬ��˳����Բ�ͬ
//////////////////////////////////////////////////////////////////////////
const static std::string TUMOR_TYPES_TREE[TUMOR_TYPE_NUM] =//For listing
{
    //1
    "Uncertain",//��ȷ���Ĳ���
    //2.	����
    "LGIEN", //�ͼ�����Ƥ������ ����
    "HGIEN", //�߼�����Ƥ������ ����
    //3
    "NO_GEIEN",//����Ƥ������
    //4
    "Uncertain_GEIEN",//��ȷ������Ƥ������
    //5 ��Ƥ������
    "PRE_LGIEN",//�ͼ�����Ƥ������ ��Ƥ������
    "PRE_HGIEN",//�߼�����Ƥ������ ��Ƥ������
    //6 ��֢
    //6.1 �ٰ�
    "AD_diablastic_comedo", //ɸ״-�۴���
    "AD_medullary", //������
    "AD_micropapillary", //΢��ͷ״�ٰ�
    "AD_slime", //�Һ�ٰ�
    "AD_zigzag", //���״�ٰ�
    "AD_signet_ring_cell",//ӡ��ϸ����
    //6.2
    "adenosquamous_carcinoma", //���۰�
    //6.3
    "Spindle-cell_carcinoma", //����ϸ����
    //6.4
    "squamous-cell_carcinoma", //��״ϸ����
    //6.5
    "undifferentiated_carcinoma", //δ�ֻ���
    //7 ���ڷ�������
    //7.1 ���ڷ����� NET
    "NET1", //NET1�����఩��
    "NET2", //NET2��
    //7.2���ڷ����� NEC
    "NEC_big_cell", //��ϸ��NEC
    "NEC_small_cell", //Сϸ��NEC
    //7.3
    "MIX_NET", //����������ڷ��ڰ�
    //7.4
    "EC-cell-serotonin-NET",//ECϸ��������5-��ɫ��NET
    //7.5
    "L-cell-secretion-glucagon-PP_PYY-NET"//Lϸ���������ȸ�Ѫ�������ĺ�PP/PYY NET
};


//////////////////////////////////////////////////////////////////////////
//���listһ��ֻ���ڵײ���չ�ñ�֤��֮ǰ�ı�ע������ݣ�����
//////////////////////////////////////////////////////////////////////////
const std::string TUMOR_TYPES[TUMOR_TYPE_NUM] =
{
    //1
    "Uncertain",//��ȷ���Ĳ���
    //2.	����
    "LGIEN", //�ͼ�����Ƥ������ ����
    "HGIEN", //�߼�����Ƥ������ ����
    //3
    "NO_GEIEN",//����Ƥ������
    //4
    "Uncertain_GEIEN",//��ȷ������Ƥ������
    //5 ��Ƥ������
    "PRE_LGIEN",//�ͼ�����Ƥ������ ��Ƥ������
    "PRE_HGIEN",//�߼�����Ƥ������ ��Ƥ������
    //6 ��֢
    //6.1 �ٰ�
    "AD_diablastic_comedo", //ɸ״-�۴���
    "AD_medullary", //������
    "AD_micropapillary", //΢��ͷ״�ٰ�
    "AD_slime", //�Һ�ٰ�
    "AD_zigzag", //���״�ٰ�
    "AD_signet_ring_cell",//ӡ��ϸ����
    //6.2
    "adenosquamous_carcinoma", //���۰�
    //6.3
    "Spindle-cell_carcinoma", //����ϸ����
    //6.4
    "squamous-cell_carcinoma", //��״ϸ����
    //6.5
    "undifferentiated_carcinoma", //δ�ֻ���
    //7 ���ڷ�������
    //7.1 ���ڷ����� NET
    "NET1", //NET1�����఩��
    "NET2", //NET2��
    //7.2���ڷ����� NEC
    "NEC_big_cell", //��ϸ��NEC
    "NEC_small_cell", //Сϸ��NEC
    //7.3
    "MIX_NET", //����������ڷ��ڰ�
    //7.4
    "EC-cell-serotonin-NET",//ECϸ��������5-��ɫ��NET
    //7.5
    "L-cell-secretion-glucagon-PP_PYY-NET"//Lϸ���������ȸ�Ѫ�������ĺ�PP/PYY NET
};

#endif // !TUMOR_TYPE_SHANGHAI_NO1_H
