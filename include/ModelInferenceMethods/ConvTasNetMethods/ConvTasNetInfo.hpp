#pragma once

#include <cstdint>
#include <type_traits>

template <int64_t BatchSize,
          int64_t BufferSize,
          int64_t OutputChannels,
          int64_t SampleRate>
struct ConvTasNetInfo {
    static constexpr int64_t batch_size() { return BatchSize; }
    static constexpr int64_t buffer_size() { return BufferSize; }
    static constexpr int64_t output_channels() { return OutputChannels; }
    static constexpr int64_t sample_rate() { return SampleRate; }
};

template <typename T>
concept IsConvTasNetInfo = requires(T t) {
    std::bool_constant<(T::batch_size(), true)>::value;
    std::bool_constant<(T::buffer_size(), true)>::value;
    std::bool_constant<(T::output_channels(), true)>::value;
    std::bool_constant<(T::sample_rate(), true)>::value;
};