#ifndef MEDIMGUTIL_MI_CONSOLE_ECHO_H
#define MEDIMGUTIL_MI_CONSOLE_ECHO_H

#include <string>
#include <map>
#include <memory>
#include <boost/thread/mutex.hpp>

#include "util/mi_util_export.h"
#include "util/mi_exception.h"
#include "util/mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

class IEchoAction {
public:
    virtual int execute() = 0;
};

class ConsoleEcho {
public:
    const static int STOP_ECHO_SIGNAL = -1;

    ConsoleEcho() {}
    virtual ~ConsoleEcho() {}

    void register_action(const std::string& command, std::shared_ptr<IEchoAction> action) {
        boost::mutex::scoped_lock locker(_mutex);
        _actions[command] = action;
    }

    void unregister_action(const std::string& command) {
        boost::mutex::scoped_lock locker(_mutex);
        auto it_action = _actions.find(command);
        if (it_action != _actions.end()) {
            _actions.erase(it_action);
        }
    }

    void run() {
        std::string command;
        while(std::cin >> command) {
            std::shared_ptr<IEchoAction> action;
            {
                boost::mutex::scoped_lock locker(_mutex);
                auto it_action = _actions.find(command);
                if (it_action != _actions.end()) {
                    action = it_action->second;
                } else {
                    MI_UTIL_LOG(MI_WARNING) << "console echo : invalid command.";
                    continue;
                }
            }

            if (action) {
                try {
                    if(STOP_ECHO_SIGNAL == action->execute()) {
                       break; 
                    }
                } catch(const Exception& e) {
                    MI_UTIL_LOG(MI_ERROR) << "console echo catch&skip exception: " << e.what();
                } catch(const std::exception& e) {
                    MI_UTIL_LOG(MI_ERROR) << "console echo catch&skip exception: " << e.what();
                }
            }
        }
    }

private:
    boost::mutex _mutex;
    std::map<std::string, std::shared_ptr<IEchoAction>> _actions;

private:
    DISALLOW_COPY_AND_ASSIGN(ConsoleEcho);
};

MED_IMG_END_NAMESPACE

#endif 