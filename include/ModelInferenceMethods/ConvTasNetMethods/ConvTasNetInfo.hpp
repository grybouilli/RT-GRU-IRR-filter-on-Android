#pragma once

#include <cstdint>
#include <type_traits>

template <int8_t  BatchSize,
          int16_t BufferSize,
          int8_t  OutputChannels,
          int     SampleRate>
struct ConvTasNetInfo {
    static constexpr int8_t  batch_size() { return BatchSize; }
    static constexpr int16_t buffer_size() { return BufferSize; }
    static constexpr int8_t  output_channels() { return OutputChannels; }
    static constexpr int     sample_rate() { return SampleRate; }
};

template <typename T>
concept IsConvTasNetInfo = requires(T t) {
    std::bool_constant<(T::batch_size(), true)>::value;
    std::bool_constant<(T::buffer_size(), true)>::value;
    std::bool_constant<(T::output_channels(), true)>::value;
    std::bool_constant<(T::sample_rate(), true)>::value;
};