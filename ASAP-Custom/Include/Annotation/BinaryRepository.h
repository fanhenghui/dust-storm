#ifndef BINARY_REPOSITORY_H_
#define BINARY_REPOSITORY_H_

#include "annotation_export.h"
#include "Repository.h"

class ANNOTATION_EXPORT BinaryRepository : public Repository
{
public:
    BinaryRepository(const std::shared_ptr<AnnotationList>& list);
    virtual ~BinaryRepository();
    virtual bool save() const;

private:
    virtual bool loadFromRepo();
protected:
private:
};

#endif // !BINARY_REPOSITORY_H_
