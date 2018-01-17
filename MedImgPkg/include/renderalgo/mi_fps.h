#ifndef MED_IMG_RENDERALGO_MI_FPS_H
#define MED_IMG_RENDERALGO_MI_FPS_H

#include "renderalgo/mi_render_algo_export.h"

MED_IMG_BEGIN_NAMESPACE

class FPS {
public: 
    FPS(): _ave_duration(0.0f), _frame(1) {

    }
    ~FPS() {}

    int fps(float delta_time_ms) {
        ++ _frame;
        if (1 == _frame) {
            _sum_duration = delta_time_ms;
            _ave_duration = delta_time_ms;
        } else {
            _sum_duration += delta_time_ms;
            _ave_duration = _sum_duration / (float)_frame;
        }

        const int FRAME_SUM = 100;
        if (_frame > FRAME_SUM) {
            _frame = 0;
            _sum_duration = 0.0f;
        }

        return (int)(1.0f / _ave_duration * 1000);
    }

    void reset() {
        _frame = 1;
        _ave_duration = 0.0f;
    }

private:
    float _ave_duration;
    float _sum_duration;
    int _frame;
};

MED_IMG_END_NAMESPACE

#endif