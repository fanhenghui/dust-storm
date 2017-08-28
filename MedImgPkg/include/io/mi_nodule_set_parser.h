#ifndef MEDIMGIO_NODULE_SET_PARSER_H
#define MEDIMGIO_NODULE_SET_PARSER_H

#include "io/mi_io_define.h"
#include "io/mi_io_export.h"
#include "mbedtls/rsa.h"

MED_IMG_BEGIN_NAMESPACE

class NoduleSet;
class IO_Export NoduleSetParser {
public:
  NoduleSetParser();
  ~NoduleSetParser();

  void set_series_id(const std::string &series_id);

  IOStatus load_as_csv(const std::string &file_path,
                       std::shared_ptr<NoduleSet> &nodule_set);
  IOStatus save_as_csv(const std::string &file_path,
                       const std::shared_ptr<NoduleSet> &nodule_set);

  IOStatus load_as_rsa_binary(const std::string &file_path,
                              const mbedtls_rsa_context &rsa,
                              std::shared_ptr<NoduleSet> &nodule_set);
  IOStatus save_as_rsa_binary(const std::string &file_path,
                              const mbedtls_rsa_context &rsa,
                              const std::shared_ptr<NoduleSet> &nodule_set);

protected:
private:
  std::string _series_id;
};

MED_IMG_END_NAMESPACE

#endif