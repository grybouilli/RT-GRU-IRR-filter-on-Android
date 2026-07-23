#pragma once

#include <sndfile.h>

#include <sndfile_utils.hpp>
#include <string>

template <typename T, int Format, int IntType = 0>
class SoundFileWriter {
   public:
    SoundFileWriter(const std::string filepath,
                    const int         sample_rate,
                    const int         channels) :
        m_info{.samplerate = sample_rate,
               .channels   = channels,
               .format     = type_to_sf_type<T, IntType>() | Format},
        m_file{sf_open(filepath.c_str(), SFM_WRITE, &m_info)} {}
    ~SoundFileWriter() { sf_close(m_file); }

    sf_count_t write(const T* data, const size_t length) {
        if constexpr (std::same_as<T, float>) {
            return sf_write_float(m_file, data, length);
        } else if constexpr (std::same_as<T, double>) {
            return sf_write_double(m_file, data, length);
        } else if constexpr (std::same_as<T, int>) {
            return sf_write_int(m_file, data, length);
        }
    }

   private:
    SF_INFO  m_info;
    SNDFILE* m_file;
};