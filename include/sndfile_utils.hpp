#pragma once

#include <sndfile.h>

#include <concepts>

template <typename T, int IntType = SF_FORMAT_PCM_24>
int type_to_sf_type() {
    if constexpr (std::same_as<T, float>) {
        return SF_FORMAT_FLOAT;
    } else if constexpr (std::same_as<T, double>) {
        return SF_FORMAT_DOUBLE;
    } else if constexpr (std::same_as<T, int>) {
        return IntType;
    }
}