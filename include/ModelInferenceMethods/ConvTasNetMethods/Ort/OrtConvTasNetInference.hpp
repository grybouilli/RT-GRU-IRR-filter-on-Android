#pragma once

#include <AudioParams.hpp>
#include <ModelInferenceMethods/ConvTasNetMethods/ConvTasNetInfo.hpp>
#include <ModelInferenceMethods/GeneralInferenceParams.hpp>
#include <ModelInferenceMethods/ModelInferenceMethodBase.hpp>
#include <ModelInferenceMethods/OrtUtils/OrtSessionHandler.hpp>
#include <ModelInferenceMethods/OrtUtils/OrtTensorBuffer.hpp>
#include <Resampler.hpp>
#include <array>

template <IsConvTasNetInfo ConvTasNet>
class OrtConvTasNetInference final
    : public ModelInferenceMethodBase<ConvTasNet> {
   public:
    OrtConvTasNetInference(const ConvTasNet&            convtasnet,
                           const GeneralInferenceParams gparams,
                           const OrtParams&             ieparams) :
        ModelInferenceMethodBase<ConvTasNet>(convtasnet, gparams, ieparams),
        m_session_handler{gparams.model_filename,
                          ieparams.EP_name,
                          gparams.debug_mode_on,
                          ieparams.EP_options,
                          ieparams.optimized_model},
        m_memory_info{
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)},
        m_binding{m_session_handler.session()},
        m_x_data{m_memory_info,
                 {convtasnet.batch_size(), convtasnet.buffer_size()}},
        m_output{m_memory_info,
                 {convtasnet.batch_size(),
                  convtasnet.output_channels(),
                  convtasnet.buffer_size()}},
        m_downsampler{(double)convtasnet.sample_rate() /
                      gparams.dsp_sample_rate},
        m_upsampler{(double)gparams.dsp_sample_rate / convtasnet.sample_rate()},
        m_selected_out_channel{0} {
        const auto Ba = convtasnet.batch_size();
        const auto Bu = convtasnet.buffer_size();
        const auto O  = convtasnet.output_channels();
    }

    bool run(float* audio, const size_t num_samples) override {
        static const size_t B = static_cast<size_t>(ConvTasNet::buffer_size());
        const auto          samples = std::min(num_samples, B);
        const size_t        expected_ds_size =
            (int)(m_downsampler.get_ratio() * samples);

        std::shift_left(m_x_data.buffer_memory.begin(),
                        m_x_data.buffer_memory.end(),
                        expected_ds_size);  // discard the oldest buffer

        auto ds_frames = m_downsampler.resample(
            audio,
            num_samples,
            m_x_data.buffer_memory.data() + m_x_data.buffer_memory.size() -
                expected_ds_size,
            expected_ds_size);

        m_binding.ClearBoundInputs();
        m_binding.BindInput("input", m_x_data.tensor);

        m_binding.ClearBoundOutputs();
        m_binding.BindOutput("output", m_output.tensor);

        m_session_handler.session().Run(Ort::RunOptions{nullptr}, m_binding);

        const auto offset = (m_selected_out_channel + 1) * B - expected_ds_size;
        std::vector<float> upsampled_voices(samples);
        auto               gen_frames =
            m_upsampler.resample(m_output.buffer_memory.data() + offset,
                                 expected_ds_size,
                                 upsampled_voices.data(),
                                 upsampled_voices.size());
        std::cout << "expected ds size: " << expected_ds_size << std::endl;
        std::cout << "downsampled frames: " << ds_frames << std::endl;
        std::cout << "offset: " << offset << std::endl;
        std::cout << "upsampled frames: " << gen_frames << std::endl;
        std::cout << "upsampled voices: " << upsampled_voices.size()
                  << std::endl;

        const auto upsampled_frames = std::min((int)samples, gen_frames);

        std::memset(
            audio,
            0,
            samples *
                sizeof(float));  // DEBUG : check if output is not just input
        for (auto sample = 0; sample < upsampled_frames; ++sample) {
            *(audio + sample) = upsampled_voices[sample];
        }
        return true;  // TODO: return the amount of treated samples
    }

    /**
     * @brief Select which output channel to output when using run
     *
     * @param channel an integer that is either 0 or 1
     */
    void select_output_channel(const size_t channel) {
        m_selected_out_channel = channel % 2;
    }

   private:
    OrtSessionHandler m_session_handler;

    Ort::MemoryInfo m_memory_info;
    Ort::IoBinding  m_binding;

    OrtTensorBuffer<float, 2> m_x_data;
    OrtTensorBuffer<float, 3> m_output;

    Resampler<1> m_downsampler;
    Resampler<2> m_upsampler;

    size_t m_selected_out_channel;
};