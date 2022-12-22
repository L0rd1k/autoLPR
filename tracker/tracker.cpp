#include "tracker.h"

bool alpr::Tracker::is_inited() {
    return isInited_;
}

void alpr::Tracker::set_inited(bool flag) {
    isInited_ = flag;
}