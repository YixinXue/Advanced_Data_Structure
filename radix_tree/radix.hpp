#include <iostream>
#include <memory>
#include <array>
#include <cstdint>

class CompressedRadixTree {
private:
    struct Node {
        uint32_t val = 0;        // 当前节点保存的 bit 段
        int len = 0;             // val 的长度
        Node* parent = nullptr;  // 父节点
        std::array<std::unique_ptr<Node>, 4> children;

        Node(Node* p = nullptr, uint32_t v = 0, int l = 0)
            : parent(p), val(v), len(l) {}
    };

    std::unique_ptr<Node> root = std::make_unique<Node>();

public:
    // 插入 value
    void insert(int32_t value) {
        Node* tmp = root.get();
        for (int i = 32; i > 0;) {
            int bit = (value >> (i - 2)) & 0x3;
            if (!tmp->children[bit]) {
                uint32_t val = ((uint32_t)value << (32 - i)) >> (32 - i);
                tmp->children[bit] = std::make_unique<Node>(tmp, val, i);
                break;
            }

            tmp = tmp->children[bit].get();
            int len = tmp->len;
            uint32_t bit2 = ((uint32_t)value << (32 - i)) >> (32 - len);

            if (bit2 != tmp->val) {
                // 计算公共前缀
                uint32_t mask = bit2 ^ tmp->val;
                int len2 = 0;
                while (mask) { mask >>= 1; ++len2; }
                if (len2 % 2 != 0) len2++;

                Node* p = tmp->parent;

                // 拆分旧节点
                auto new_child = std::make_unique<Node>(p, tmp->val >> len2, len - len2);
                for (int j = 0; j < 4; ++j) {
                    new_child->children[j] = std::move(tmp->children[j]);
                    if (new_child->children[j])
                        new_child->children[j]->parent = new_child.get();
                }

                tmp->parent = new_child.get();
                tmp->val &= (1u << len2) - 1; // 保留低 len2 位
                tmp->len = len2;

                int bias = i - len + len2;
                uint32_t val = ((uint32_t)value << (32 - bias)) >> (32 - bias);
                new_child->children[val >> (len2 - 2)] = std::make_unique<Node>(new_child.get(), val, bias);

                p->children[bit] = std::move(new_child);
                break;
            }

            i -= len;
        }
    }

    // 查找 value
    bool find(int32_t value) const {
        Node* tmp = root.get();
        for (int i = 32; i > 0;) {
            int bit = (value >> (i - 2)) & 0x3;
            if (!tmp->children[bit]) return false;
            tmp = tmp->children[bit].get();
            int len = tmp->len;
            uint32_t bit2 = ((uint32_t)value << (32 - i)) >> (32 - len);
            if (bit2 != tmp->val) return false;
            i -= len;
        }
        return true;
    }

    // 删除 value
    bool remove(int32_t value) {
        Node* tmp = root.get();
        for (int i = 32; i > 0;) {
            int bit = (value >> (i - 2)) & 0x3;
            if (!tmp->children[bit]) return false;
            tmp = tmp->children[bit].get();
            int len = tmp->len;
            uint32_t bit2 = ((uint32_t)value << (32 - i)) >> (32 - len);
            if (bit2 != tmp->val) return false;
            i -= len;
        }

        Node* p = tmp->parent;
        if (!p) return false; // root 不删除
        int idx = -1;
        for (int i = 0; i < 4; ++i)
            if (p->children[i] && p->children[i].get() == tmp)
                idx = i;

        if (idx != -1) {
            p->children[idx].reset(); // 自动释放内存

            // 如果父节点只剩一个孩子，合并路径
            int cnt = 0, child_idx = -1;
            for (int i = 0; i < 4; ++i)
                if (p->children[i]) { ++cnt; child_idx = i; }

            if (cnt == 1) {
                auto& child = p->children[child_idx];
                p->val = (p->val << child->len) | child->val;
                p->len += child->len;
                for (int i = 0; i < 4; ++i) {
                    if (child->children[i]) {
                        p->children[i] = std::move(child->children[i]);
                        p->children[i]->parent = p;
                    }
                }
            }
        }
        return true;
    }
};
