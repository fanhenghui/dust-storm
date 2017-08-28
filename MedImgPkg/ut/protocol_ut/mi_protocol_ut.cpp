#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include "mi_message.pb.h"

using namespace medical_imaging;
int main(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::cout << "test protobuf\n";
  chdir(dirname(argv[0]));

  MsgPoint2 pt2;
  pt2.set_x(100);
  pt2.set_y(200);

  char *data = new char[pt2.ByteSize()];
  int size = pt2.ByteSize();
  if (!pt2.SerializeToArray(data, size)) {
    std::cout << "serialize failed!\n";
  }

  MsgPoint2 pt22;
  if (!pt22.ParseFromArray(data, size)) {
    std::cout << "parse failed!\n";
  } else {
    std::cout << pt22.x() << " " << pt22.y() << std::endl;
  }

  return 0;
}