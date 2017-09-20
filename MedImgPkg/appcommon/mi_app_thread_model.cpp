#include "mi_app_thread_model.h"

#include "boost/thread/condition.hpp"
#include "boost/thread/thread.hpp"

#include "glresource/mi_gl_context.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_scene_base.h"
#include "util/mi_file_util.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_message_queue.h"

#include "mi_app_cell.h"
#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_operation_interface.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

struct AppThreadModel::InnerThreadData {
    boost::thread _th;
    boost::mutex _mutex;
    boost::condition _condition;
};

struct AppThreadModel::InnerQueue {
    MessageQueue<std::shared_ptr<IOperation>> _msg_queue;
};

AppThreadModel::AppThreadModel()
    : _rendering(false), _sending(false), _th_rendering(new InnerThreadData()),
      _th_sending(new InnerThreadData()), _th_operating(new InnerThreadData()),
      _op_queue(new InnerQueue()) {
    // Creare gl context
    UIDType uid(0);
    _glcontext = GLResourceManagerContainer::instance()
                 ->get_context_manager()
                 ->create_object(uid);
    _glcontext->initialize();
    _glcontext->create_shared_context(RENDERING_CONTEXT);
    _glcontext->create_shared_context(OPERATION_CONTEXT);

    _op_queue->_msg_queue.activate();
}

AppThreadModel::~AppThreadModel() {}

std::shared_ptr<GLContext> AppThreadModel::get_gl_context() {
    return _glcontext;
}

void AppThreadModel::set_client_proxy(std::shared_ptr<IPCClientProxy> proxy) {
    _proxy = proxy;
}

void AppThreadModel::set_controller(std::shared_ptr<AppController> controller) {
    _controller = controller;
}

void AppThreadModel::push_operation(const std::shared_ptr<IOperation>& op) {
    _op_queue->_msg_queue.push(op);
}

void AppThreadModel::pop_operation(std::shared_ptr<IOperation>* op) {
    _op_queue->_msg_queue.pop(op);
}

void AppThreadModel::start() {
    try {
        _th_operating->_th =
            boost::thread(boost::bind(&AppThreadModel::process_operating, this));

        _th_sending->_th =
            boost::thread(boost::bind(&AppThreadModel::process_sending, this));

        _th_rendering->_th =
            boost::thread(boost::bind(&AppThreadModel::process_rendering, this));
    } catch (...) {
        MI_APPCOMMON_LOG(MI_FATAL) << "app thread model start failed.";
        APPCOMMON_THROW_EXCEPTION("app thread model start failed.");
    }
}

void AppThreadModel::stop() {
    _th_rendering->_th.interrupt();
    _th_rendering->_condition.notify_one();

    _th_sending->_th.interrupt();
    _th_sending->_condition.notify_one();

    _th_operating->_th.interrupt();
    _th_operating->_condition.notify_one();

    _th_rendering->_th.join();
    _th_sending->_th.join();
    _th_operating->_th.join();

    _op_queue->_msg_queue.deactivate();
}

void AppThreadModel::process_operating() {   
    try{ 
        while(true) {
            std::shared_ptr<IOperation> op;
            this->pop_operation(&op);

            boost::mutex::scoped_lock locker(_th_rendering->_mutex);

            try {
                int err = op->execute();
                if (-1 == err) {
                    MI_APPCOMMON_LOG(MI_ERROR) << "op execute failed.";
                    continue;
                }
            } catch (const Exception& e) {
                MI_APPCOMMON_LOG(MI_ERROR) << "operating run failed with exception: " << e.what() << "(SKIP IT FOR KEEPTING RUNNING)";
                continue;
            } catch (const std::exception& e) {
                MI_APPCOMMON_LOG(MI_ERROR) << "operating run failed with exception: " << e.what() << "(SKIP IT FOR KEEPTING RUNNING)";
                continue;
            }

            // interrupt point
            boost::this_thread::interruption_point();
            _rendering = true;
            _th_rendering->_condition.notify_one();
        }
    }  catch (boost::thread_interrupted& e) {
        MI_APPCOMMON_LOG(MI_INFO) << "operating thread is interrupted.";;
        throw e;
    } catch (...) {
        MI_APPCOMMON_LOG(MI_FATAL) << "operating run failed with unknow exception";
        throw;
    }
}

void AppThreadModel::process_rendering() {
    try {
        while (true) {
            std::deque<unsigned int> dirty_cells;
            std::deque<std::shared_ptr<SceneBase>> dirty_scenes;

            _glcontext->make_current(RENDERING_CONTEXT);
            ///\ 1 render
            {
                boost::mutex::scoped_lock locker(_th_rendering->_mutex);

                while (!_rendering) {
                    _th_rendering->_condition.wait(_th_rendering->_mutex);
                }

                // render all dirty cells
                std::shared_ptr<AppController> controller = _controller.lock();
                APPCOMMON_CHECK_NULL_EXCEPTION(controller);

                std::map<unsigned int, std::shared_ptr<AppCell>> cells =
                            controller->get_cells();

                for (auto it = cells.begin(); it != cells.end(); ++it) {
                    std::shared_ptr<SceneBase> scene = it->second->get_scene();
                    APPCOMMON_CHECK_NULL_EXCEPTION(scene);

                    if (scene->get_dirty()) {
                        dirty_cells.push_back(it->first);
                        dirty_scenes.push_back(scene);
                        scene->render();
                        scene->set_dirty(false);
                    }
                }
                // interrupt point
                boost::this_thread::interruption_point();
                _rendering = false;
            }

            /// \2 get image result to buffer
            // download all dirty scene image to buffer
            for (auto it = dirty_scenes.begin(); it != dirty_scenes.end(); ++it) {
                (*it)->download_image_buffer();
            }

            // tell sending the change and swap dirty scene image buffer
            {
                boost::mutex::scoped_lock dirty_cells_locker(_dirty_cells_mutex);
                _dirty_cells = dirty_cells;

                for (auto it = dirty_scenes.begin(); it != dirty_scenes.end(); ++it) {
                    (*it)->swap_image_buffer();
                }
            }
            _sending = true;
            _th_sending->_condition.notify_one();

            _glcontext->make_noncurrent();
        }
    } catch (const Exception& e) {
        MI_APPCOMMON_LOG(MI_FATAL) << "rendering run failed with exception: " << e.what();
        throw e;
    } catch (const std::exception& e) {
        MI_APPCOMMON_LOG(MI_FATAL) << "rendering run failed with exception: " << e.what();
        throw e;
    } catch (boost::thread_interrupted& e) {
        MI_APPCOMMON_LOG(MI_INFO) << "rendering thread is interrupted.";
    } catch (...) {
        MI_APPCOMMON_LOG(MI_FATAL) << "rendering run failed failed with unknow exception";
        throw;
    }
}

void AppThreadModel::process_sending() {
    try {
        while (true) {
            ///\ sending image to fe by pic proxy
            boost::mutex::scoped_lock locker(_th_sending->_mutex);

            while (!_sending) {
                _th_sending->_condition.wait(_th_sending->_mutex);
            }

            // get dirty cells to be sending
            std::deque<unsigned int> dirty_cells;
            {
                boost::mutex::scoped_lock dirty_cells_locker(_dirty_cells_mutex);
                dirty_cells = _dirty_cells;
            }

            std::shared_ptr<AppController> controller = _controller.lock();
            APPCOMMON_CHECK_NULL_EXCEPTION(controller);

            // sendong image buffer
            for (auto it = dirty_cells.begin(); it != dirty_cells.end(); ++it) {
                const unsigned int cell_id = *it;
                std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
                APPCOMMON_CHECK_NULL_EXCEPTION(cell);
                std::shared_ptr<SceneBase> scene = cell->get_scene();
                APPCOMMON_CHECK_NULL_EXCEPTION(scene);
                int width(32), height(32);
                scene->get_display_size(width, height);

                IPCDataHeader header;
                header._sender = static_cast<unsigned int>(controller->get_local_pid());
                header._receiver =
                    static_cast<unsigned int>(controller->get_server_pid());
                ;
                header._msg_id = COMMAND_ID_BE_SEND_IMAGE;
                header._msg_info0 = cell_id;
                header._msg_info1 = 0;
                header._data_type = 0;
                header._big_end = 0;

                unsigned char* buffer = nullptr;
                int buffer_size = 0;
                scene->get_image_buffer(buffer, buffer_size);
                APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
                header._data_len = static_cast<unsigned int>(buffer_size);

                std::cout << "send data length : "
                          << static_cast<unsigned int>(buffer_size) << std::endl;

                // Testing code write image to disk
                //{
                //    std::stringstream ss;
                //    ss << "/home/wangrui22/data/img_buffer_cell_" << cell_id << ".jpeg";
                //    FileUtil::write_raw(ss.str(), buffer, buffer_size);
                //}

                _proxy->async_send_message(header, (char*)buffer);
            }

            // interrupt point
            boost::this_thread::interruption_point();
            _sending = false;
        }
    } catch (const Exception& e) {
        MI_APPCOMMON_LOG(MI_FATAL) << "sending run failed with exception: " << e.what();
        throw e;
    } catch (const std::exception& e) {
        MI_APPCOMMON_LOG(MI_FATAL) << "sending run failed with exception: " << e.what();
        throw e;
    } catch (boost::thread_interrupted& e) {
        MI_APPCOMMON_LOG(MI_INFO) << "sending thread is interrupted.";
    } catch (...) {
        MI_APPCOMMON_LOG(MI_FATAL) << "sending run failed failed with unknow exception";
        throw;
    }
}

MED_IMG_END_NAMESPACE