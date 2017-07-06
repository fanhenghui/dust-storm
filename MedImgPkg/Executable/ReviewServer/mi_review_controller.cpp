#include "mi_review_controller.h"

#include "MedImgRenderAlgorithm/mi_volume_infos.h"

MED_IMG_BEGIN_NAMESPACE

ReviewController::ReviewController()
{

}

ReviewController::~ReviewController()
{

}

void ReviewController::initialize()
{
    
}

void ReviewController::set_volume_infos(std::shared_ptr<VolumeInfos> volumeinfos)
{
    _volumeinfos = volumeinfos;
}

std::shared_ptr<VolumeInfos> ReviewController::get_volume_infos()
{
    return _volumeinfos;
}




MED_IMG_END_NAMESPACE