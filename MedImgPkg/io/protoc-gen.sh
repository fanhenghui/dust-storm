protoc --proto_path=../config/protobuf/ --cpp_out=./ ../config/protobuf/mi_message.proto
mv mi_message.pb.h ../include/io/