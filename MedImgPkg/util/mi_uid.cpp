#include "mi_uid.h"
#include "uuid/uuid.h"

MED_IMG_BEGIN_NAMESPACE

std::string UUIDGenerator::uuid() {
    uuid_t uu;
    char buf[1024];
    uuid_generate(uu);
    uuid_unparse(uu,buf);
    std::string ss;
    return std::string(buf);
}

MED_IMG_END_NAMESPACE