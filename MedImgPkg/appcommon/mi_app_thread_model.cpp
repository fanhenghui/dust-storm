#include "mi_app_thread_model.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_operation_interface.h"

#include "glresource/mi_gl_context.h"
#include "glresource/mi_gl_resource_manager_container.h"

#include "renderalgo/mi_scene_base.h"

#include "mi_app_cell.h"
#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_app_none_image_interface.h"

MED_IMG_BEGIN_NAMESPACE

AppThreadModel::AppThreadModel(): _rendering(false), _sending(false)  {
    // Creare gl context
    UIDType uid(0);
    _glcontext = GLResourceManagerContainer::instance()
                 ->get_context_manager()
                 ->create_object(uid);
    _glcontext->initialize();
    _glcontext->create_shared_context(RENDERING_CONTEXT);
    _glcontext->create_shared_context(OPERATION_CONTEXT);

    _op_msg_queue.activate();
}

AppThreadModel::~AppThreadModel() {}

std::shared_ptr<GLContext> AppThreadModel::get_gl_context() {
    return _glcontext;
}

void AppThreadModel::set_client_proxy(std::shared_ptr<IPCClientProxy> proxy) {
    _client_proxy = proxy;
}

void AppThreadModel::set_controller(std::shared_ptr<AppController> controller) {
    _controller = controller;
}

void AppThreadModel::push_operation(const std::shared_ptr<IOperation>& op) {
    _op_msg_queue.push(op);
}

void AppThreadModel::pop_operation(std::shared_ptr<IOperation>* op) {
    _op_msg_queue.pop(op);
}

void AppThreadModel::start(const std::string& unix_path) {
    try {
        _thread_operating = boost::thread(boost::bind(&AppThreadModel::process_operating, this));
        _thread_sending = boost::thread(boost::bind(&AppThreadModel::process_sending, this));
        _thread_rendering = boost::thread(boost::bind(&AppThreadModel::process_rendering, this));

        _thread_dbs_recving = boost::thread(boost::bind(&AppThreadModel::process_dbs_recving, this));

        _client_proxy->set_path(unix_path);
        _client_proxy->run();
    } catch (...) {
        MI_APPCOMMON_LOG(MI_FATAL) << "app thread model start failed.";
        APPCOMMON_THROW_EXCEPTION("app thread model start failed.");
    }
}

void AppThreadModel::stop() {
    _thread_rendering.interrupt();
    _condition_rendering.notify_one();

    _thread_sending.interrupt();
    _condition_sending.notify_one();

    _thread_operating.interrupt();

    _thread_rendering.join();
    _thread_sending.join();
    _thread_operating.join();

    _op_msg_queue.deactivate();

    _client_proxy_dbs->stop();
    _thread_dbs_recving.join();
}

void AppThreadModel::process_operating() {   
    try{ 
        while(true) {
            std::shared_ptr<IOperation> op;
            this->pop_operation(&op);

            boost::mutex::scoped_lock locker(_mutex_rendering);

            try {
                int err = op->execute();
                op->reset();//release ipc data
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
            _condition_rendering.notify_one();
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
            std::deque<unsigned int> dirty_images;
            std::deque<unsigned int> dirty_none_images;
            std::deque<std::shared_ptr<SceneBase>> dirty_scenes;

            _glcontext->make_current(RENDERING_CONTEXT);
            {
                ///\ 1 render
                boost::mutex::scoped_lock locker(_mutex_rendering);

                while (!_rendering) {
                    _condition_rendering.wait(_mutex_rendering);
                }

                // render all dirty cells
                std::shared_ptr<AppController> controller = _controller.lock();
                APPCOMMON_CHECK_NULL_EXCEPTION(controller);

                // GL resource update (discard)
                GLResourceManagerContainer::instance()->update_all();

                std::map<unsigned int, std::shared_ptr<AppCell>> cells =
                            controller->get_cells();

                for (auto it = cells.begin(); it != cells.end(); ++it) {
                    std::shared_ptr<SceneBase> scene = it->second->get_scene();
                    APPCOMMON_CHECK_NULL_EXCEPTION(scene);
                    if (scene->get_dirty()) {
                        dirty_images.push_back(it->first);
                        dirty_scenes.push_back(scene);
                        scene->render();
                        scene->set_dirty(false);
                    }

                    std::shared_ptr<IAppNoneImage> none_image = it->second->get_none_image();
                    if(none_image && none_image->check_dirty()) {
                        dirty_none_images.push_back(it->first);
                        none_image->update();
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
            _glcontext->make_noncurrent();

            // tell sending the change and swap dirty scene image buffer
            {
                boost::mutex::scoped_lock dirty_images_locker(_dirty_images_mutex);
                _dirty_images = dirty_images;

                for (auto it = dirty_scenes.begin(); it != dirty_scenes.end(); ++it) {
                    (*it)->swap_image_buffer();
                }
            }
            {
                boost::mutex::scoped_lock dirty_none_images_locker(_dirty_none_images_mutex);
                _dirty_none_images = dirty_none_images;
            }
            _sending = true;
            _condition_sending.notify_one();
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
            boost::mutex::scoped_lock locker(_mutex_sending);

            while (!_sending) {
                _condition_sending.wait(_mutex_sending);
            }

            std::shared_ptr<AppController> controller = _controller.lock();
            APPCOMMON_CHECK_NULL_EXCEPTION(controller);

            // get dirty scenes to be sending
            std::deque<unsigned int> dirty_images;
            {
                boost::mutex::scoped_lock dirty_images_locker(_dirty_images_mutex);
                dirty_images = _dirty_images;
            }
            // sendong image buffer
            for (auto it = dirty_images.begin(); it != dirty_images.end(); ++it) {
                const unsigned int cell_id = *it;
                std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
                APPCOMMON_CHECK_NULL_EXCEPTION(cell);
                std::shared_ptr<SceneBase> scene = cell->get_scene();
                APPCOMMON_CHECK_NULL_EXCEPTION(scene);
                int width(32), height(32);
                scene->get_display_size(width, height);

                IPCDataHeader header;
                header.sender = static_cast<unsigned int>(controller->get_local_pid());
                header.receiver = static_cast<unsigned int>(controller->get_server_pid());
                header.msg_id = COMMAND_ID_FE_BE_SEND_IMAGE;
                header.cell_id = cell_id;
                header.op_id = 0;
                header.reserved0 = 0;
                header.reserved1 = 0;

                unsigned char* buffer = nullptr;
                int buffer_size = 0;
                scene->get_image_buffer(buffer, buffer_size);
                APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
                header.data_len = static_cast<unsigned int>(buffer_size);

                MI_APPCOMMON_LOG(MI_TRACE) << "send image data length: " << buffer_size;

                // Testing code write image to disk
                // {
                //     int w,h;
                //     scene->get_display_size(w,h);
                //     std::stringstream ss;
                //     ss << "/home/wangrui22/data/img_buffer_cell_" << cell_id << "_" << w << "_" << h <<".jpeg";
                //     FileUtil::write_raw(ss.str(), buffer, buffer_size);
                // }

                _client_proxy->sync_send_data(header, (char*)buffer);
            }

            // get dirty cells to be sending
            std::deque<unsigned int> dirty_none_images;
            {
                boost::mutex::scoped_lock dirty_none_images_locker(_dirty_none_images_mutex);
                dirty_none_images = _dirty_none_images;
            }
            for (auto it = dirty_none_images.begin(); it != dirty_none_images.end(); ++it) {
                const unsigned int cell_id = *it;
                std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
                APPCOMMON_CHECK_NULL_EXCEPTION(cell);
                std::shared_ptr<IAppNoneImage> none_image = cell->get_none_image();
                APPCOMMON_CHECK_NULL_EXCEPTION(none_image);

                IPCDataHeader header;
                header.sender = static_cast<unsigned int>(controller->get_local_pid());
                header.receiver = static_cast<unsigned int>(controller->get_server_pid());
                header.msg_id = COMMAND_ID_FE_BE_SEND_NONE_IMAGE;
                header.cell_id = cell_id;
                header.op_id = 0;
                header.reserved0 = 0;
                header.reserved1 = 0;

                int buffer_size = 0;
                char* buffer = none_image->serialize_dirty(buffer_size);
                if (nullptr == buffer || buffer_size == 0) {
                    //MI_APPCOMMON_LOG(MI_WARNING) << "dirty none image has no serialized dirty buffer.";
                    continue;
                }
                header.data_len = static_cast<unsigned int>(buffer_size);
                MI_APPCOMMON_LOG(MI_TRACE) << "send none image data length: " << buffer_size;
                _client_proxy->sync_send_data(header, buffer);
                if (nullptr != buffer) {
                    delete [] buffer;
                }
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

void AppThreadModel::set_client_proxy_dbs(std::shared_ptr<IPCClientProxy> proxy) {
    _client_proxy_dbs = proxy;
}

void AppThreadModel::process_dbs_recving() {
    _client_proxy_dbs->run();
}

MED_IMG_END_NAMESPACE