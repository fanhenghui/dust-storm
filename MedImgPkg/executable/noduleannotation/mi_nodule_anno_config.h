#ifndef NODULE_ANNO_MI_NODULE_ANNO_CONFIG_H
#define NODULE_ANNO_MI_NODULE_ANNO_CONFIG_H

#include <map>
#include <string>
#include "boost/thread/mutex.hpp"

enum PreSetWLType {
    CT_ABDOMEN,
    CT_LUNGS,
    CT_BRAIN,
    CT_ANGIO,
    CT_BONE,
    CT_CHEST,
    CT_Preset_ALL,
};

class NoduleAnnoConfig {
public:
    static NoduleAnnoConfig* instance();
    ~NoduleAnnoConfig();

    void bind_config_file(const std::string& file);

    void initialize();
    void finalize();

    void set_nodule_file_rsa(bool b);
    bool get_nodule_file_rsa();

    void set_preset_wl(PreSetWLType type, float ww, float wl);
    void get_preset_wl(PreSetWLType type, float& ww, float& wl) const;

    void set_last_open_direction(const std::string& path);
    std::string get_last_open_direction() const;

    int get_double_click_interval() const;

protected:
private:
    NoduleAnnoConfig();
    void init();

    static NoduleAnnoConfig* _s_instance;
    static boost::mutex _s_mutex;

private:
    std::string _config_file;
    bool _is_nodule_file_rsa;
    std::map<PreSetWLType , std::pair<float , float>> _preset_windowing;
    std::string _last_open_direction;
    int _double_click_interval; 
};

#endif