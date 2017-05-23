#pragma once

#include <memory>
#include <vector>
#include <mutex>

#include "mine.h"
#include "genetic_alg.h"

typedef void (*RenderCallback)(void) ;

class NeuralNet;
class GeneticAlg;
class MineSweeper;
class GLTexture2D;
class BattleSpace
{
public:
    BattleSpace();
    ~BattleSpace();

    void render();

    void update();

    void register_render_callback(RenderCallback callback);

    void enable_fast_training(bool flag);
    bool is_fast_training() const { return _fast_training; };

private:
    void load_resource_i();

    void create_start_battle_space_i();

private:
    std::vector<std::shared_ptr<MineSweeper>> _mine_sweeper;
    std::vector<Mine> _mines;
    std::shared_ptr<GeneticAlg> _genetic_alg;

    std::shared_ptr<GLTexture2D> _tex_mine_sweeper;
    std::shared_ptr<GLTexture2D> _tex_mine_sweeper2;
    std::shared_ptr<GLTexture2D> _tex_mine;

    std::vector<Chromosome> _weights_collection;

    RenderCallback _render_callback;

    std::mutex _mutex;

    bool _fast_training;
};

