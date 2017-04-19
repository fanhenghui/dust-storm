#ifndef MED_IMAGING_MODEL_VOI_H_
#define MED_IMAGING_MODEL_VOI_H_

#include "MedImgCommon/mi_model_interface.h"
#include "MedImgIO/mi_voi.h"
#include <list>

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export VOIModel : public IModel
{
public:
    VOIModel();
    virtual ~VOIModel();
    void AddVOISphere(const VOISphere& voi);
    VOISphere GetVOISphere(int id);
    void ModifyVOISphereName(int id , std::string sName);
    void ModifyVOISphereDiameter(int id , double dDiameter);
    void ModifyVOISphereRear(const VOISphere& voi);
    void RemoveVOISphere(int id);
    const std::list<VOISphere>& GetVOISpheres() const;
    void GetVOISpheres(std::list<VOISphere>& l) const;
protected:
private:
    std::list<VOISphere> m_VOISphereList;
};

MED_IMAGING_END_NAMESPACE

#endif