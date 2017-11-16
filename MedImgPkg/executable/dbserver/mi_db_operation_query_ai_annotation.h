#ifndef MED_IMG_MI_DB_OPERATION_QUERY_AI_ANNOTATION_H
#define MED_IMG_MI_DB_OPERATION_QUERY_AI_ANNOTATION_H

#include "mi_db_operation.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryAIAnnotation: public DBOperation {
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