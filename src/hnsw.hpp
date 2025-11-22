#pragma once
#include <unordered_set>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include "../util/util.hpp"

static int global_id = 0;

class Node {
public:
    int global_id;
    int *data;
    int label;
    int max_level;
    std::vector<std::unordered_set<Node*>> neighbors;

    Node(int *data, int label, int max_level = 0) {
        this->global_id = global_id++;
        this->data = data;
        this->label = label;
        this->max_level = max_level;
        neighbors.resize(max_level + 1);
    }

    void set_neighbors(std::unordered_set<Node*> nbs, int level) {
        neighbors[level] = nbs;
    }

    void add_neighbor(Node* n, int level) {
        neighbors[level].insert(n);
    }
};

class HNSW {
public:
    Node* entry_point;
    int M;          // max neighbors per node
    int M_max;      // max neighbors for upper layers
    int ef_construction;
    int ef_search;
    int vec_dim;
    int max_level;

    HNSW(int dim=128) {
        entry_point = nullptr;
        M = 30;
        M_max = 30;
        ef_construction = 100;
        ef_search = 50;
        vec_dim = dim;
        max_level = 0;
    }

    void insert(int *data, int label);
    std::vector<int> query(int *data, int k);
    std::unordered_set<Node*> search_layer(int *q, Node* ep, int ef, int layer);
    std::unordered_set<Node*> select_neighbors(int *q, std::unordered_set<Node*> candidates, int num);
    Node* get_nearest(std::unordered_set<Node*> candidates, int *q);
    Node* get_furthest(std::unordered_set<Node*> candidates, int *q);
};

void HNSW::insert(int *data, int label) {
    if (!entry_point) {
        entry_point = new Node(data, label, 0);
        return;
    }

    Node* ep = entry_point;
    int L = HNSWLab::get_random_level();
    Node* new_node = new Node(data, label, L);

    // 上层搜索
    for (int layer = max_level; layer > L; layer--) {
        auto W = search_layer(data, ep, 1, layer);
        ep = get_nearest(W, data);
    }

    // 从 min(L, max_level) 层开始插入
    for (int layer = std::min(L, max_level); layer >= 0; layer--) {
        auto W = search_layer(data, ep, ef_construction, layer);
        auto neighbors = select_neighbors(data, W, M);

        new_node->set_neighbors(neighbors, layer);
        for (auto n : neighbors) {
            n->add_neighbor(new_node, layer);
            if (n->neighbors[layer].size() > M_max) {
                n->neighbors[layer].erase(get_furthest(n->neighbors[layer], n->data));
            }
        }

        ep = get_nearest(W, data);
    }

    if (L > max_level) {
        max_level = L;
        entry_point = new_node;
    }
}

std::vector<int> HNSW::query(int *data, int k) {
    std::vector<int> res;
    if (!entry_point) return res;

    Node* ep = entry_point;
    for (int layer = max_level; layer > 0; layer--) {
        auto W = search_layer(data, ep, 1, layer);
        ep = get_nearest(W, data);
    }

    auto W = search_layer(data, ep, ef_search, 0);
    auto neighbors = select_neighbors(data, W, k);

    for (auto n : neighbors)
        res.push_back(n->label);

    return res;
}

std::unordered_set<Node*> HNSW::search_layer(int *q, Node* ep, int ef, int layer) {
    std::unordered_set<Node*> W, V, C;
    W.insert(ep); V.insert(ep); C.insert(ep);

    while (!C.empty()) {
        Node* c = get_nearest(C, q);
        C.erase(c);
        Node* f = get_furthest(W, q);

        if (HNSWLab::l2distance(c->data, q, vec_dim) > HNSWLab::l2distance(f->data, q, vec_dim))
            break;

        for (auto n : c->neighbors[layer]) {
            if (V.find(n) == V.end()) {
                V.insert(n);
                f = get_furthest(W, q);
                if (W.size() < ef || HNSWLab::l2distance(n->data, q, vec_dim) < HNSWLab::l2distance(f->data, q, vec_dim)) {
                    W.insert(n);
                    C.insert(n);
                    if (W.size() > ef)
                        W.erase(f);
                }
            }
        }
    }
    return W;
}

std::unordered_set<Node*> HNSW::select_neighbors(int *q, std::unordered_set<Node*> candidates, int num) {
    auto comp = [&](std::pair<long, Node*> a, std::pair<long, Node*> b){ return a.first > b.first; };
    std::priority_queue<std::pair<long, Node*>, std::vector<std::pair<long, Node*>>, decltype(comp)> pq(comp);

    for (auto n : candidates)
        pq.push({HNSWLab::l2distance(n->data, q, vec_dim), n});

    std::unordered_set<Node*> res;
    for (int i = 0; i < num && !pq.empty(); i++) {
        res.insert(pq.top().second);
        pq.pop();
    }
    return res;
}

Node* HNSW::get_nearest(std::unordered_set<Node*> candidates, int *q) {
    Node* nearest = nullptr;
    long mind = LONG_MAX;
    for (auto n : candidates) {
        long d = HNSWLab::l2distance(n->data, q, vec_dim);
        if (d < mind) {
            mind = d;
            nearest = n;
        }
    }
    return nearest;
}

Node* HNSW::get_furthest(std::unordered_set<Node*> candidates, int *q) {
    Node* furthest = nullptr;
    long maxd = LONG_MIN;
    for (auto n : candidates) {
        long d = HNSWLab::l2distance(n->data, q, vec_dim);
        if (d > maxd) {
            maxd = d;
            furthest = n;
        }
    }
    return furthest;
}
