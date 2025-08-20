#include "help_cpp.h"

int retrieve_py(const std::vector<float>& tree, int capacity, float s) {
    int idx = 0;
    while (idx < capacity - 1) {
        int left = 2 * idx + 1;
        if (s <= tree[left]) {
            idx = left;
        } else {
            s -= tree[left];
            idx = left + 1;
        }
    }
    return idx;
}