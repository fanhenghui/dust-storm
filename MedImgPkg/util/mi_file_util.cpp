#include "mi_file_util.h"
#include "mi_util_logger.h"

#include <fstream>
#include "boost/filesystem.hpp"

MED_IMG_BEGIN_NAMESPACE

namespace {
void get_files(const std::string& root , const std::set<std::string>& postfix,
               std::vector<std::string>& files) {
    if (root.empty()) {
        return ;
    } else {
        std::vector<std::string> dirs;

        for (boost::filesystem::directory_iterator it(root) ;
                it != boost::filesystem::directory_iterator() ; ++it) {
            if (boost::filesystem::is_directory(*it)) {
                dirs.push_back(it->path().filename().string());
            } else {
                const std::string ext = boost::filesystem::extension(*it);

                if (postfix.empty()) {
                    files.push_back(root + std::string("/") + it->path().filename().string());
                } else {
                    for (auto itext = postfix.begin(); itext != postfix.end(); ++itext) {
                        if (*itext == ext) {
                            files.push_back(root + std::string("/") + it->path().filename().string());
                            break;
                        }
                    }
                }
            }
        }

        for (unsigned int i = 0; i < dirs.size(); ++i) {
            const std::string next_dir(root + "/" + dirs[i]);
            get_files(next_dir , postfix , files);
        }

    }
}

void get_files(const std::string& root , const std::set<std::string>& postfix,
    std::map<std::string , std::vector<std::string>>& files) {
    if (root.empty()) {
        return ;
    } else {
        std::vector<std::string> dirs;

        for (boost::filesystem::directory_iterator it(root) ;
            it != boost::filesystem::directory_iterator() ; ++it) {
                if (boost::filesystem::is_directory(*it)) {
                    dirs.push_back(it->path().filename().string());
                } else {
                    const std::string ext = boost::filesystem::extension(*it);
                    if (postfix.find(ext) != postfix.end())
                    {
                        if (files.find(ext) == files.end()) {
                            std::vector<std::string> file(1 , root + std::string("/") + it->path().filename().string());
                            files[ext] = file;
                        } else {
                            files[ext].push_back(root + std::string("/") + it->path().filename().string());
                        }
                    }
                }
        }

        for (unsigned int i = 0; i < dirs.size(); ++i) {
            const std::string next_dir(root + "/" + dirs[i]);
            get_files(next_dir , postfix , files);
        }

    }
}

}

void FileUtil::get_all_file_recursion(
    const std::string& root ,
    const std::set<std::string>& postfix ,
    std::vector<std::string>& files) {
    if (root.empty()) {
        return;
    }

    get_files(root , postfix, files);
}

void FileUtil::get_all_file_recursion( 
    const std::string& root , 
    const std::set<std::string>& postfix , 
    std::map<std::string , std::vector<std::string>>& files) {
    if (root.empty() || postfix.empty()){
        return;
    }

    get_files(root , postfix , files);
}

int FileUtil::write_raw(const std::string& path , void* buffer , unsigned int length) {
    if (nullptr == buffer) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::write_raw input buffer is null.";
        return -1;
    }

    if (path.empty()) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::write_raw input path is empty.";
    }

    std::ofstream out(path.c_str() , std::ios::out | std::ios::binary);

    if (!out.is_open()) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::write_raw open file " << path << " failed.";
        return -1;
    }

    out.write((char*)buffer , length);
    out.close();
    MI_UTIL_LOG(MI_TRACE) << "FileUtil::write_raw in file " << path << " success.";
    return 0;
}

int FileUtil::read_raw(const std::string& path, void* buffer, unsigned int length) {
    if (nullptr == buffer) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::read_raw input buffer is null.";
        return -1;
    }

    if (path.empty()) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::read_raw input path is empty.";
    }

    std::ifstream in(path.c_str() , std::ios::in | std::ios::binary);

    if (!in.is_open()) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::read_raw open file " << path << " failed.";
        return -1;
    }

    in.seekg(0, std::ios::end);
    unsigned int file_size = static_cast<unsigned int>(in.tellg());
    if (file_size < length) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::read_raw input file " << path << 
        " file size(" << file_size << ") less than input buffer length(" << length << ").";
        in.close();
        return -1;
    }
    if (file_size > length) {
        MI_UTIL_LOG(MI_WARNING) << "FileUtil::read_raw input file " << path << 
        " file size(" << file_size << ") large than input buffer length(" << length << ").";
    }

    in.seekg(0, std::ios::beg);
    in.read((char*)(buffer), length);
    in.close();

    return 0;
}

int FileUtil::read_raw_ext(const std::string& path, char*& buffer, unsigned int& length) {
    if (path.empty()) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::read_raw input path is empty.";
    }

    std::ifstream in(path.c_str() , std::ios::in | std::ios::binary);

    if (!in.is_open()) {
        MI_UTIL_LOG(MI_ERROR) << "FileUtil::read_raw open file " << path << " failed.";
        return -1;
    }

    in.seekg(0, std::ios::end);
    length = static_cast<unsigned int>(in.tellg());
    char* buffer_ = new char[length];
    buffer = buffer_;

    in.seekg(0, std::ios::beg);
    in.read(buffer_, length);
    in.close();

    return 0;
}

MED_IMG_END_NAMESPACE