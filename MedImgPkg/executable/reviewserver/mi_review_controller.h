#ifndef MED_IMG_REVIEW_CONTROLLER_H_
#define MED_IMG_REVIEW_CONTROLLER_H_

#include "util/mi_ipc_client_proxy.h"

#include <memory>
#include <string>

#include "appcommon/mi_app_controller.h"

MED_IMG_BEGIN_NAMESPACE

class VolumeInfos;
class ReviewController : public AppController {
public:
  ReviewController();
  virtual ~ReviewController();

  virtual void initialize();

  void set_volume_infos(std::shared_ptr<VolumeInfos> volumeinfos);
  std::shared_ptr<VolumeInfos> get_volume_infos();

protected:
private:
  std::shared_ptr<VolumeInfos> _volumeinfos;
};

MED_IMG_END_NAMESPACE

#endif