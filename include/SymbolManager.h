#ifndef INCLUDE_SYMBOLMANAGER_H_
#define INCLUDE_SYMBOLMANAGER_H_

#include <map>

class SymbolManager {
 public:
    static SymbolManager* getInstance();
    int getDictionarySize() const {
        return dict_.size();
    }
    size_t getIndexOfSymbol(char c) const {
        return dict_.at(c);
    }
    int getIndexOfOutput(size_t c) {
        return out_map_.at(c);
    }

 private:
    SymbolManager();
    static SymbolManager* manager_;
    std::map<char, int> dict_;
    std::map<size_t, int> out_map_;
};

#endif
