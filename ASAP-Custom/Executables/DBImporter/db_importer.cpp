#include <iostream>
#include <sstream>
#include <fstream>

#include "boost/filesystem.hpp"

//mysql begin
#include "mysql_connection.h"
#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/resultset.h"
#include "cppconn/statement.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/sqlstring.h"
//mysql end

#include "IO/MultiResolutionImageInterface/MultiResolutionImageReader.h"
#include "IO/MultiResolutionImageInterface/MultiResolutionImage.h"

static std::ofstream out_log;

class LogSheild
{
public:
    LogSheild()
    {
        out_log.open("db_importer.log", std::ios::out);
        if (out_log.is_open())
        {
            out_log << "DB importer log:\n";
        }
    }
    ~LogSheild()
    {
        out_log.close();
    }
protected:
private:
};

static void get_all_files(const std::string& root, unsigned int& num , std::vector<std::string>& file_names , std::vector<std::string>& file_paths)
{
    if (root.empty())
    {
        return;
    }
    else
    {
        std::vector<std::string> dirs;
        for (boost::filesystem::directory_iterator it(root); it != boost::filesystem::directory_iterator(); ++it)
        {
            if (boost::filesystem::is_directory(*it))
            {
                dirs.push_back(it->path().filename().string());
            }
            else
            {
                const std::string ext = boost::filesystem::extension(*it);
                //////////////////////////////////////////////////////////////////////////
                //Formats from http://openslide.org/
                //Aperio(.svs, .tif)
                //Hamamatsu(.vms, .vmu, .ndpi)
                //Leica(.scn)
                //MIRAX(.mrxs)
                //Philips(.tiff)
                //Sakura(.svslide)
                //Trestle(.tif)
                //Ventana(.bif, .tif)
                //Generic tiled TIFF(.tif)
                //////////////////////////////////////////////////////////////////////////
                if (ext == ".tif" || ext == ".svs" 
                    || ext == ".vms" || ext == ".vmu" || ext == ".ndpi"
                    || ext == ".scn" 
                    || ext == ".mrxs"
                    || ext == ".tiff"
                    || ext == "svslide"
                    || ext == ".bif")
                {
                    file_names.push_back(it->path().filename().string());
                    file_paths.push_back(root + "/" + it->path().filename().string());
                    ++num;
                }
            }
        }

        for (unsigned int i = 0; i < dirs.size(); ++i)
        {
            const std::string next_dir(root + "/" + dirs[i]);

            get_all_files(next_dir, num , file_names , file_paths);
        }
    }
}

/*单个字符转十六进制字符串，长度增大2被*/
static void char_to_hex(unsigned char ch, char *szHex)
{
    int i;
    unsigned char byte[2];
    byte[0] = ch / 16;
    byte[1] = ch % 16;
    for (i = 0; i < 2; i++)
    {
        if (byte[i] >= 0 && byte[i] <= 9)
            szHex[i] = '0' + byte[i];
        else
            szHex[i] = 'a' + byte[i] - 10;
    }
    szHex[2] = 0;
}

/*字符串转换函数，中间调用上面的函数*/
void char_str_to_hex_str(char *pucCharStr, int iSize, char *pszHexStr)
{
    int i;
    char szHex[3];
    pszHexStr[0] = 0;
    for (i = 0; i < iSize; i++)
    {
        char_to_hex(pucCharStr[i], szHex);
        strcat(pszHexStr, szHex);
    }
}

static bool get_md5(const std::string& file, unsigned char(&md5)[16], MultiResolutionImageReader& img_reader)
{
    MultiResolutionImage* img = img_reader.open(file);
    if (!img)
    {
        out_log << "ERROR : file : " << file << " open failed!\n";
        return false;
    }

    if (!img->valid())
    {
        out_log << "ERROR : file : " << file << " open valid!\n";
        return false;
    }

    if (!img->getImgHash(md5))
    {
        out_log << "ERROR : calculate file : " << file << " md5 failed!\n";
        return false;
    }

    return true;
}


int main(int argc , char* argv[])
{
    LogSheild log_sheild;
    //exe "ip" "username" "password "database" "file_root"
    if (argc != 6)
    {
        out_log << "ERROR : invalid input.\n";
        out_log << "\tFormat : ip username password database file_root\n";
        return -1;
    }

    //Get all files
    const std::string ip = std::string("tcp://") + std::string(argv[1]);
    const std::string user_name = argv[2];
    const std::string password = argv[3];
    const std::string database = argv[4];
    std::string file_root = argv[5];
    for (int i = 0; i < file_root.size() ; ++i)
    {
        if (file_root[i] == '\\')
        {
            file_root[i] = '/';
        }
    }

    out_log << "IP : " << ip << std::endl;
    out_log << "User name : " << user_name << std::endl;
    out_log << "Password : " << password << std::endl;
    out_log << "Database : " << database << std::endl;
    out_log << "File root : " << file_root << std::endl;

    std::vector<std::string> file_names;
    std::vector<std::string> file_paths;
    unsigned int file_num = 0;
    get_all_files(file_root, file_num, file_names , file_paths);
    if (file_num != file_names.size() || file_num != file_paths.size())
    {
        out_log << "ERROR : Get files error!\n";
        return -1;
    }
    if (file_paths.empty())
    {
        out_log << "ERROR : Empty files!\n";
        return -1;
    }

    out_log << "File to be import list : \n";
    for (unsigned int i = 0 ; i< file_num ; ++i)
    {
        out_log << "\t" << file_paths[i] << std::endl;
    }
    out_log << std::endl;

    //////////////////////////////////////////////////////////////////////////
    //SQL 
    //connect database
    sql::Connection *con = nullptr;
    try
    {
        //create connect
        sql::Driver *driver = get_driver_instance();
        con = driver->connect(ip.c_str(), user_name.c_str(), password.c_str());
        //con = driver->connect("tcp://127.0.0.1:3306", "root", "0123456");
        con->setSchema(database.c_str());

        sql::Statement *stmt = con->createStatement();
        delete stmt;

        MultiResolutionImageReader imgReader;
        unsigned char md5[16];
        for (unsigned int i = 0; i< file_num ; ++i)
        {
            //1 calculate md5
            if (!get_md5(file_paths[i] , md5, imgReader))
            {
                continue;
            }
            //convert md5 to 32 hex char
            char md5_hex[16 * 2];
            char_str_to_hex_str((char*)md5, 16, md5_hex);

            //2 insert into database
            std::stringstream ss;
            ss << "INSERT INTO images (name , md5 , path) values (\'";
            ss << file_names[i] << "\' , \'";
            for (int i = 0; i < 32; ++i)
            {
                ss << md5_hex[i];
            }
            ss << "\' , \'";
            ss << file_paths[i] << "\');";

            const std::string insert_sql = ss.str();

            sql::PreparedStatement *pstmt = con->prepareStatement(insert_sql.c_str());
            sql::ResultSet *res = pstmt->executeQuery();
            delete pstmt;
            pstmt = nullptr;
            delete res;
            res = nullptr;
        }

        delete con;
        con = nullptr;
    }
    catch (const sql::SQLException& e)
    {
        out_log << "ERROR : ";
        out_log<< "# ERR: SQLException in " << __FILE__;
        out_log<< "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        out_log<< "# ERR: " << e.what();
        out_log<< " (MySQL error code: " << e.getErrorCode();
        out_log<< ", SQLState: " << e.getSQLState() << " )" << std::endl;

        delete con;
        con = nullptr;

        return -1;
    }

    out_log << "Import to database success.\n";
    return 0;
}