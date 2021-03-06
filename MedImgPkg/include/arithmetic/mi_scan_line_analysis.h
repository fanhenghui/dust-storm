#ifndef MED_IMG_SCAN_LINE_ANALYSIS_H
#define MED_IMG_SCAN_LINE_ANALYSIS_H

#include "arithmetic/mi_arithmetic_export.h"

MED_IMG_BEGIN_NAMESPACE

template<class T>
class FillStrategy {
public:
    FillStrategy() {};
    virtual ~FillStrategy() {};
    virtual void fill(int x, int y , int width , int height , T* img) = 0;
protected:
private:
};

template<class T>
class ScanLineAnalysis {
public:
    struct Pt2 {
        int x, y;

        Pt2(int xx , int yy): x(xx), y(yy) {

        }

        Pt2() {

        }
    };

    struct Edge {
        float x;//corss X
        float dx;//edge's slope_r : delta_x / delta_y
        int ymax;//edge's top y
        Edge* next;

        Edge(): x(0), dx(0), ymax(0), next(nullptr) {

        }
    };

public:
    ScanLineAnalysis() {};
    ~ScanLineAnalysis() {};

    void fill(T* img , int width , int height ,
              const std::vector<Pt2>& pts , T label = 1) {
        ARITHMETIC_CHECK_NULL_EXCEPTION(img);

        if (width < 2 || height < 2) {
            ARITHMETIC_THROW_EXCEPTION("Invalid size of input image!");
        }

        if (pts.empty()) {
            ARITHMETIC_THROW_EXCEPTION("empty contour");
        }

        //1 create new edge table (net)
        Edge** net = new Edge*[height];

        for (int i = 0; i < height ; ++i) {
            net[i] = new Edge();//initialize head
        }

        for (size_t i = 0 ; i < pts.size() - 1 ; ++i) {
            const bool comp = pts[i].y < pts[i + 1].y;
            const Pt2& pt0 = comp ? pts[i] : pts[i + 1];
            const Pt2& pt1 = comp ? pts[i + 1] : pts[i];

            if (pt0.y == pt1.y) {
                continue;//discard horizational line
            }

            Edge* new_edge = new Edge();
            new_edge->x = static_cast<float>(pt0.x);
            new_edge->dx = static_cast<float>(pt1.x - pt0.x) / static_cast<float>(pt1.y - pt0.y);
            new_edge->ymax = pt1.y;

            Edge* cur_edge = net[pt0.y];

            while (cur_edge->next != nullptr) {
                cur_edge = cur_edge->next;
            }

            cur_edge->next = new_edge;
        }

        //last point connect with the first
        {
            const bool comp = pts[pts.size() - 1].y < pts[0].y;
            const Pt2& pt0 = comp ? pts[pts.size() - 1] : pts[0];
            const Pt2& pt1 = comp ? pts[0] : pts[pts.size() - 1];

            if (pt0.y != pt1.y) { //discard horizational line
                Edge* new_edge = new Edge();
                new_edge->x = static_cast<float>(pt0.x);
                new_edge->dx = static_cast<float>(pt1.x - pt0.x) / static_cast<float>(pt1.y - pt0.y);
                new_edge->ymax = pt1.y;

                Edge* cur_edge = net[pt0.y];

                while (cur_edge->next != nullptr) {
                    cur_edge = cur_edge->next;
                }

                cur_edge->next = new_edge;
            }


        }

        //2 create active edge table (aet)
        Edge** aet = new Edge*[height];

        for (int i = 0; i < height ; ++i) {
            aet[i] = new Edge();//initialize head
        }

        for (int i = 0 ; i < height ; ++i) {
            Edge* net_edge = net[i]->next;

            while (nullptr != net_edge) { //inset net to aet
                const int ymin = i;
                const int ymax = net_edge->ymax;
                const float x = net_edge->x;
                const float dx = net_edge->dx;

                for (int j = ymin ; j < ymax ; ++j) { //not include ymax(floor)
                    Edge* new_edge = new Edge();
                    new_edge->ymax = ymax;
                    new_edge->dx  = dx;
                    new_edge->x = x + dx * (j - ymin);

                    //add to AET (sort by x)
                    Edge* aet_edge = aet[j];

                    if (nullptr == aet_edge->next) { //add  firstly
                        aet_edge->next = new_edge;
                    } else {
                        Edge* pre = aet_edge;
                        aet_edge = aet_edge->next;

                        while (nullptr != aet_edge && aet_edge->x < new_edge->x) {
                            pre = aet_edge;
                            aet_edge = aet_edge->next;
                        }

                        pre->next = new_edge;
                        new_edge->next = aet_edge;
                    }
                }

                net_edge = net_edge->next;
            }
        }

        //3 fill between 2 point
        for (int i = 0 ; i < height ; ++i) {
            Edge* aet_edge = aet[i]->next;

            if (nullptr == aet_edge) {
                continue;
            }

            while (nullptr != aet_edge) {
                if (nullptr == aet_edge->next) {
                    break;
                }

                float x0 = aet_edge->x;
                float x1 = aet_edge->next->x;

                int x00 = static_cast<int>(x0);
                int x11 = static_cast<int>(x1);

                for (int x = x00 ; x <= x11 ; ++x) {
                    img[i * width  + x ] = label;
                }

                aet_edge = aet_edge->next->next;
            }
        }

        //Delete NET AET
        for (int i = 0; i < height ; ++i) {
            Edge* edge = net[i];

            while (nullptr == edge) {
                Edge* to_be_delete = edge;
                edge = edge->next;
                delete to_be_delete;
            }

            edge = aet[i];

            while (nullptr == edge) {
                Edge* to_be_delete = edge;
                edge = edge->next;
                delete to_be_delete;
            }
        }

    };

    void fill(T* img , int width , int height ,
              const std::vector<Pt2>& pts , FillStrategy<T>* strategy) {
        ARITHMETIC_CHECK_NULL_EXCEPTION(strategy);

        ARITHMETIC_CHECK_NULL_EXCEPTION(img);

        if (width < 2 || height < 2) {
            ARITHMETIC_THROW_EXCEPTION("Invalid size of input image!");
        }

        if (pts.empty()) {
            ARITHMETIC_THROW_EXCEPTION("empty contour");
        }

        //1 create new edge table (net)
        Edge** net = new Edge*[height];

        for (int i = 0; i < height ; ++i) {
            net[i] = new Edge();//initialize head
        }

        for (int i = 0 ; i < pts.size() - 1 ; ++i) {
            const bool comp = pts[i].y < pts[i + 1].y;
            const Pt2& pt0 = comp ? pts[i] : pts[i + 1];
            const Pt2& pt1 = comp ? pts[i + 1] : pts[i];

            if (pt0.y == pt1.y) {
                continue;//discard horizational line
            }

            Edge* new_edge = new Edge();
            new_edge->x = static_cast<float>(pt0.x);
            new_edge->dx = static_cast<float>(pt1.x - pt0.x) / static_cast<float>(pt1.y - pt0.y);
            new_edge->ymax = pt1.y;

            Edge* cur_edge = net[pt0.y];

            while (cur_edge->next != nullptr) {
                cur_edge = cur_edge->next;
            }

            cur_edge->next = new_edge;
        }

        //last point connect with the first
        {
            const bool comp = pts[pts.size() - 1].y < pts[0].y;
            const Pt2& pt0 = comp ? pts[pts.size() - 1] : pts[0];
            const Pt2& pt1 = comp ? pts[0] : pts[pts.size() - 1];

            if (pt0.y != pt1.y) { //discard horizational line
                Edge* new_edge = new Edge();
                new_edge->x = static_cast<float>(pt0.x);
                new_edge->dx = static_cast<float>(pt1.x - pt0.x) / static_cast<float>(pt1.y - pt0.y);
                new_edge->ymax = pt1.y;

                Edge* cur_edge = net[pt0.y];

                while (cur_edge->next != nullptr) {
                    cur_edge = cur_edge->next;
                }

                cur_edge->next = new_edge;
            }


        }

        //2 create active edge table (aet)
        Edge** aet = new Edge*[height];

        for (int i = 0; i < height ; ++i) {
            aet[i] = new Edge();//initialize head
        }

        for (int i = 0 ; i < height ; ++i) {
            Edge* net_edge = net[i]->next;

            while (nullptr != net_edge) { //inset net to aet
                const int ymin = i;
                const int ymax = net_edge->ymax;
                const float x = net_edge->x;
                const float dx = net_edge->dx;

                for (int j = ymin ; j < ymax ; ++j) { //not include ymax(floor)
                    Edge* new_edge = new Edge();
                    new_edge->ymax = ymax;
                    new_edge->dx  = dx;
                    new_edge->x = x + dx * (j - ymin);

                    //add to AET (sort by x)
                    Edge* aet_edge = aet[j];

                    if (nullptr == aet_edge->next) { //add  firstly
                        aet_edge->next = new_edge;
                    } else {
                        Edge* pre = aet_edge;
                        aet_edge = aet_edge->next;

                        while (nullptr != aet_edge && aet_edge->x < new_edge->x) {
                            pre = aet_edge;
                            aet_edge = aet_edge->next;
                        }

                        pre->next = new_edge;
                        new_edge->next = aet_edge;
                    }
                }

                net_edge = net_edge->next;
            }
        }

        //3 fill between 2 point
        for (int i = 0 ; i < height ; ++i) {
            Edge* aet_edge = aet[i]->next;

            if (nullptr == aet_edge) {
                continue;
            }

            while (nullptr != aet_edge) {
                if (nullptr == aet_edge->next) {
                    break;
                }

                float x0 = aet_edge->x;
                float x1 = aet_edge->next->x;

                int x00 = static_cast<int>(x0);
                int x11 = static_cast<int>(x1);

                for (int x = x00 ; x <= x11 ; ++x) {
                    strategy->fill(x , i , width , height, img);
                    //img[i*width  + x ] = label;
                }

                aet_edge = aet_edge->next->next;
            }
        }

        //Delete NET AET
        for (int i = 0; i < height ; ++i) {
            Edge* edge = net[i];

            while (nullptr == edge) {
                Edge* to_be_delete = edge;
                edge = edge->next;
                delete to_be_delete;
            }

            edge = aet[i];

            while (nullptr == edge) {
                Edge* to_be_delete = edge;
                edge = edge->next;
                delete to_be_delete;
            }
        }
    }

};

MED_IMG_END_NAMESPACE



#endif