#include "mi_file_util.h"

#include "boost/filesystem.hpp"

MED_IMG_BEGIN_NAMESPACE

namespace
{
    void get_files(const std::string& root , const std::vector<std::string>& postfix, std::vector<std::string>& files)
    {
        if (root.empty()){
            return ;
        }
        else{
            std::vector<std::string> dirs;
            for (boost::filesystem::directory_iterator it(root) ; it != boost::filesystem::directory_iterator() ; ++it ){
                if(boost::filesystem::is_directory(*it)){
                    dirs.push_back(it->path().filename().string());
                }
                else{
                    const std::string ext = boost::filesystem::extension(*it);
                    if(postfix.empty()){
                        files.push_back(it->path().filename().string());
                    }
                    else{
                        for(auto itext = postfix.begin() ; itext != postfix.end() ; ++itext){
                            if(*itext == ext){
                                files.push_back(it->path().filename().string());
                                break;
                            }
                        }
                    }
                }
            }
    
            for (unsigned int i = 0 ; i< dirs.size() ; ++i){
                const std::string next_dir(root + "/" + dirs[i]);
                get_files(next_dir , postfix , files);
            }
    
        }
    }
}

void FileUtil::get_all_file_recursion(
    const std::string& root ,
    const std::vector<std::string>& postfix , 
    std::vector<std::string>& files)
{
    if(root.empty()){
        return;
    }

    get_files(root , postfix, files);
}

MED_IMG_END_NAMESPACE