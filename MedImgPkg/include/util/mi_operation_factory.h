#ifndef MEDIMG_UTIL_MI_OPERATION_FACTORY_H
#define MEDIMG_UTIL_MI_OPERATION_FACTORY_H

#include "util/mi_util_export.h"
#include "util/mi_operation_interface.h"

#include <boost/thread/mutex.hpp>

MED_IMG_BEGIN_NAMESPACE

class Util_Export OperationFactory {
public:
    static OperationFactory* instance();
    ~OperationFactory();

    std::shared_ptr<IOperation> get_operation(unsigned int id);
    void register_operation(unsigned int id , std::shared_ptr<IOperation> op);

private:
    OperationFactory();

private:
    static OperationFactory* _s_instance;
    static boost::mutex _mutex;

    std::map<unsigned int , std::shared_ptr<IOperation>> _ops;
};

MED_IMG_END_NAMESPACE

#endif