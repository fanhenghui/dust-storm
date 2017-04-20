#pragma once

#ifndef MED_IMAGING_NAMESPACE
#define MED_IMAGING_NAMESPACE                 medical_imaging
#endif

#ifndef MED_IMAGING_BEGIN_NAMESPACE
#define MED_IMAGING_BEGIN_NAMESPACE           \
    namespace MED_IMAGING_NAMESPACE           {    /* begin namespace medical_imaging */
#endif
#ifndef MED_IMAGING_END_NAMESPACE
#define MED_IMAGING_END_NAMESPACE             }    /* end namespace medical_imaging   */
#endif


//���ھ��ȣ�����Peter Shirley�Ľ��飬���ھ��ȽϸߵĹ���-���� �ཻ���㣬������˫���ȸ�������������ɫ����������õ����ȸ��������㡣
//���Ƕ���ҽѧӦ����˵������ѡ��˫���Ȼ���Ӻ��ʣ�����ҪCPU���ٵĵط���ת��Ϊ�����ȣ��ٽ�������
typedef double Real;

#define DOUBLE_EPSILON 1e-15
#define FLOAT_EPSILON 1e-6f