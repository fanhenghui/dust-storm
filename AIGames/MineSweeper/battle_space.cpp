#include "battle_space.h"

#include <chrono>

#include "Core/gl_texture_2d.h"
#include "Core/gl_resource_manager_container.h"
#include "Core/gl_utils.h"
#include "Core/targa.h"

#include "param.h"
#include "neural_net.h"
#include "mine_sweeper.h"
#include "genetic_alg.h"


BattleSpace::BattleSpace():_render_callback(nullptr), _fast_training(false)
{
    _genetic_alg.reset(new GeneticAlg(
        Param::_crossover_rate, Param::_mutation_rate, Param::_max_perturbation));

    load_resource_i();

    create_start_battle_space_i();
}

BattleSpace::~BattleSpace()
{

}

void BattleSpace::render()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    for (auto it = _mines.begin(); it != _mines.end(); ++it)
    {
        //std::unique_lock<std::mutex> locker(_mutex);
        (*it).draw();
    }

    for (auto it = _mine_sweeper.begin() ; it != _mine_sweeper.end(); ++it)
    {
        (*it)->draw();
    }

    glPopMatrix();
    glPopAttrib();
}

void BattleSpace::update()
{
    //1 Run tick_num times to calculate fitness score
    for (int i = 0 ; i< Param::_ticks_num ; ++i)
    {
        auto start = std::chrono::system_clock::now();

        for (int j = 0 ; j<Param::_mine_sweeper_num ; ++j)
        {
            //One sweeper(one neural net)
            int grab_mine_id = _mine_sweeper[j]->update(_mines);
            if (grab_mine_id > 0)
            {
                //std::cout << "Got it \n";
                //std::unique_lock<std::mutex> locker(_mutex);

                _mines[grab_mine_id].reset();
                _mine_sweeper[j]->increment_fitness(1.0);
            }
        }
        if (!_fast_training)
        {
            if (_render_callback)
            {
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                while ((double)(duration.count()) < 1000.0 / Param::_update_fps)
                {
                    end = std::chrono::system_clock::now();
                    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                }
                _render_callback();
            }
        }
    }

    //GA to train ANN weight
    _weights_collection.clear();
    _weights_collection.reserve(Param::_mine_sweeper_num);
    for (int i = 0; i < Param::_mine_sweeper_num; ++i)
    {
        _weights_collection.push_back(_mine_sweeper[i]->to_chromosome());
    }

    //////////////////////////////////////////////////////////////////////////
    //Add weight splits to consider one neuron's weight as crossover unit(thus don't cut one neuron's weight when crossover)
    if (Param::_neuron_as_crossover_unit ==1 &&  0 == _genetic_alg->get_generation())
    {
        _genetic_alg->set_crossover_splits(_mine_sweeper[0]->get_weight_splits());
    }
    //////////////////////////////////////////////////////////////////////////

    _weights_collection = _genetic_alg->epoch(_weights_collection);
    unsigned int elite = _genetic_alg->get_fittest_score();
    for (int i = 0 ; i<Param::_mine_sweeper_num ; ++i)
    {
        _mine_sweeper[i]->set_weights(_weights_collection[i]._weights);
        _mine_sweeper[i]->reset();
    }
    for (int i = 0; i<Param::_elite_num ; ++i)
    {
        _mine_sweeper[i]->i_am_elite(true);
    }
    std::cout << "Generation : " << _genetic_alg->get_generation() << " \n";
    std::cout << "Best fitness: " << _genetic_alg->get_fittest_score() << "\n";
    std::cout << "Average fitness: " << _genetic_alg->get_average_score() << "\n\n";
}

void BattleSpace::register_render_callback(RenderCallback callback)
{
    _render_callback = callback;
}

void BattleSpace::enable_fast_training(bool flag)
{
    _fast_training = flag;
}

void BattleSpace::load_resource_i()
{
    UIDType id;
    _tex_mine_sweeper = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(id);
    _tex_mine_sweeper2 = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(id);
    _tex_mine = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(id);

    int w(120), h(120), component;
    GLenum format;
    unsigned char* img_array = (unsigned char*)Targa::ReadTGABits("../../Resource/tank_1.tga", w, h, component, format);
    glEnable(GL_TEXTURE_2D);
    _tex_mine_sweeper->initialize();
    _tex_mine_sweeper->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _tex_mine_sweeper->load(GL_RGBA8, 120, 120, GL_BGRA, GL_UNSIGNED_BYTE, img_array);
    _tex_mine_sweeper->unbind();

    delete[] img_array;
    img_array = (unsigned char*)Targa::ReadTGABits("../../Resource/tank_3.tga", w, h, component, format);
    glEnable(GL_TEXTURE_2D);
    _tex_mine_sweeper2->initialize();
    _tex_mine_sweeper2->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _tex_mine_sweeper2->load(GL_RGBA8, 120, 120, GL_BGRA, GL_UNSIGNED_BYTE, img_array);
    _tex_mine_sweeper2->unbind();

    delete[] img_array;
    img_array = (unsigned char*)Targa::ReadTGABits("../../Resource/mine.tga", w, h, component, format);
    _tex_mine->initialize();
    _tex_mine->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _tex_mine->load(GL_RGBA8, 120, 120, GL_BGRA, GL_UNSIGNED_BYTE, img_array);
    _tex_mine->unbind();
    delete[] img_array;
}

void BattleSpace::create_start_battle_space_i()
{
    _mine_sweeper.reserve(Param::_mine_sweeper_num);
    for (int i = 0; i < Param::_mine_sweeper_num; ++i)
    {
        _mine_sweeper.push_back(
            std::shared_ptr<MineSweeper>(new MineSweeper()));
        _mine_sweeper[i]->set_texture(_tex_mine_sweeper, _tex_mine_sweeper2);
    }


    _mines.reserve(Param::_mines_num);
    for (int i = 0; i < Param::_mines_num; ++i)
    {
        _mines.push_back(Mine());
        _mines[i].set_texture(_tex_mine);
    }
}