#ifndef INCLUDE_VARIABLEMANAGER_H_
#define INCLUDE_VARIABLEMANAGER_H_

#include <cstdlib>
#include <vector>

class VariableManager {
 public:
    VariableManager() {
        clear();
    }
    std::vector<char> getUnused(const size_t& num);
    void clear();

 private:
    size_t first_free_;
    std::vector<char> letters_;
};

#endif  // INCLUDE_VARIABLEMANAGER_H_
