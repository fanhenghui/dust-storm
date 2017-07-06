#ifndef MED_IMG_APP_COMMON_OPERATION_FACTORY_H_
#define MED_IMG_APP_COMMON_OPERATION_FACTORY_H_

#include "MedImgAppCommon/mi_app_common_export.h"
#include "MedImgAppCommon/mi_operation_interface.h"

#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OperationFactory
{
public:
    static OperationFactory* instance();
    ~OperationFactory();

    // void register_operation(unsigned int id , std::shared_ptr<IOperation> operation);
    std::shared_ptr<IOperation> get_operation(unsigned int id);

private:
    OperationFactory();

private:
    static OperationFactory* _s_instance;
    static boost::mutex _mutex;

private:
};

MED_IMG_END_NAMESPACE

#endif