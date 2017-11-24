#ifndef MED_IMG_MI_DB_OPERATION_QUERY_AI_ANNOTATION_H
#define MED_IMG_MI_DB_OPERATION_QUERY_AI_ANNOTATION_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryAIAnnotation: public IOperation {
public:
    DBOpQueryAIAnnotation();
    virtual ~DBOpQueryAIAnnotation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpQueryAIAnnotation>(new DBOpQueryAIAnnotation());
    }
};

MED_IMG_END_NAMESPACE

#endif