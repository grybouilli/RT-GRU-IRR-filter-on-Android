#pragma once

#include <samplerate.h>

#include <array>
#include <cstring>

template <size_t Channels>
class Resampler {
   public:
    Resampler(const double ratio) :
        m_resampler(src_new(SRC_SINC_FASTEST, Channels, &m_err)),
        m_ratio{ratio} {}

    Resampler(const Resampler&)            = delete;
    Resampler& operator=(const Resampler&) = delete;

    ~Resampler() { src_delete(m_resampler); }

    template <size_t NS, size_t NO>
    int resample(const std::array<float, NS>& src,
                 std::array<float, NO>&       output) {
        SRC_DATA data;
        data.data_in       = src.data();
        data.input_frames  = NS;
        data.data_out      = output.data();
        data.output_frames = NO;
        data.src_ratio     = m_ratio;
        data.end_of_input  = data.input_frames == src.size() ? 1 : 0;

        auto res = src_process(m_resampler, &data);
        if (res != 0) return res;
        return data.output_frames_gen;
    }

    int resample(const float* src,
                 const size_t src_samples,
                 float*       output,
                 const size_t out_samples) {
        SRC_DATA data;
        data.data_in       = src;
        data.input_frames  = src_samples;
        data.data_out      = output;
        data.output_frames = out_samples;
        data.src_ratio     = m_ratio;
        data.end_of_input  = 1;

        auto res = src_process(m_resampler, &data);
        if (res != 0) return res;
        return data.output_frames_gen;
    }

    const auto get_ratio() const { return m_ratio; }

   private:
    SRC_STATE*   m_resampler;
    const double m_ratio;
    int          m_err;
};