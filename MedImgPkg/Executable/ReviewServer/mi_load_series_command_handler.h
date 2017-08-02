#ifndef MED_IMG_LOAD_SERIES_COMMAND_HANDLER_H
#define MED_IMG_LOAD_SERIES_COMMAND_HANDLER_H

#include "mi_review_common.h"
#include "MedImgUtil/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class ReviewController;
class LoadSeriesCommandHandler : public ICommandHandler
{
public:
    LoadSeriesCommandHandler(std::shared_ptr<ReviewController> controller);

    virtual ~LoadSeriesCommandHandler();

    virtual int handle_command(const IPCDataHeader& datahaeder , char* buffer);

private:
    std::weak_ptr<ReviewController> _controller;

};

MED_IMG_END_NAMESPACE

#endif