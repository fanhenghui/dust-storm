#ifndef MED_IMG_REVIEW_SERVER_H
#define MED_IMG_REVIEW_SERVER_H

#include "mi_review_common.h"
#include <string>
#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

class ReviewConfig
{
public:
    static ReviewConfig* instance();
    ~ReviewConfig();

    std::string get_test_data_root() const;

private:
    ReviewConfig();
    static ReviewConfig* _instance;
    static boost::mutex _mutex;

private:
    std::string _test_data_root;
};


MED_IMG_END_NAMESPACE


#endif