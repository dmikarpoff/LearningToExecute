#include "../include/SymbolManager.h"

SymbolManager* SymbolManager::manager_ = NULL;

SymbolManager* SymbolManager::getInstance() {
    if (manager_ == NULL)
        manager_ = new SymbolManager();
    return manager_;
}

SymbolManager::SymbolManager() {
    int last_idx = 1;
    for (char c = '0'; c <= '9'; ++c, ++last_idx)
        dict_[c] = last_idx;
    for (char c = 'a'; c <= 'z'; ++c, ++last_idx)
        dict_[c] = last_idx;
    dict_[' '] = last_idx++;
    dict_['#'] = last_idx++;
    dict_['@'] = last_idx++;
    dict_['.'] = last_idx++;
    dict_[':'] = last_idx++;
    dict_['+'] = last_idx++;
    dict_['-'] = last_idx++;
    dict_['*'] = last_idx++;
    dict_['('] = last_idx++;
    dict_[')'] = last_idx++;
    dict_['<'] = last_idx++;
    dict_['>'] = last_idx++;
    dict_['='] = last_idx++;
    dict_[0] = 0;
    last_idx = 0;
    for (char c= '0'; c <= '9'; ++c, ++last_idx)
        out_map_[dict_[c]] = c - '0';
    out_map_[dict_['.']] = 10;
}
