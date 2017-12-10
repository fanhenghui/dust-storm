protoc --proto_path=../executable/webviewer/public/data/ --cpp_out=./ ../executable/webviewer/public/data/mi_message.proto
mv mi_message.pb.h ../include/io/