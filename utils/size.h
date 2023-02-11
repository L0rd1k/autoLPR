#pragma once

namespace alpr {

template <typename T>
class Size {
public:
    Size() = default;

    Size(T width, T height)
        : width_(width),
          height_(height) {
    }

    Size(const Size& sz)
        : width_(sz.getWidth()),
          height_(sz.getHeight()) {
    }

    Size(Size&& sz)
        : width_(sz.getWidth()),
          height_(sz.getHeight()) {
    }

    Size& operator=(Size&& sz) noexcept {
        width_ = std::move(sz.getWidth());
        height_ = std::move(sz.getHeight());
        return *this;
    }

    Size operator=(const Size& sz) {
        Size new_sz;
        new_sz.setHeigth(sz.getHeight());
        new_sz.setWidth(sz.getWidth());
        return new_sz;
    }

    T getWidth() {
        return width_;
    }

    void setWidth(T width) {
        width_ = width;
    }

    T getHeight() {
        return height_;
    }

    void setHeigth(T height) {
        height_ = height;
    }

private:
    T width_, height_;
};

}