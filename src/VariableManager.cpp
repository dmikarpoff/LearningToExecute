#include <algorithm>

#include "../include/VariableManager.h"

namespace {
    int custom_random(int i) {
        return rand() % i;
    }
}  // namespace

void VariableManager::clear() {
    first_free_ = 0;
    letters_.clear();
    letters_.resize(26);
    for (char i = 'a'; i <= 'z'; ++i)
        letters_[i - 'a'] = i;
    std::random_shuffle(letters_.begin(), letters_.end(), custom_random);
}

std::vector<char> VariableManager::getUnused(const size_t &num) {
   std::vector<char> res = std::vector<char>(letters_.begin() + first_free_,
                                             letters_.begin() + first_free_ + num);
   first_free_ += num;
   return res;
}
