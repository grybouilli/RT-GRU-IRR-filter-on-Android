#include <sndfile.h>

#include <ModelInferenceMethods/ConvTasNetMethods/ConvTasNetInfo.hpp>
#include <ModelInferenceMethods/ConvTasNetMethods/Ort/OrtConvTasNetInference.hpp>
#include <ModelInferenceMethods/GeneralInferenceParams.hpp>
#include <npy.hpp>
#include <parsing_utils.hpp>

int main(int argc, char** argv) {
    std::cout << "Verbose version" << std::endl;
    GeneralInferenceParams            gparams;
    OrtParams                         ort_params;
    ConvTasNetInfo<1, 24000, 2, 8000> model;

    auto options = get_options();
    options.add_options()("file",
                          "Input file for offline inference",
                          cxxopts::value<std::string>());
    auto args = options.parse(argc, argv);

    fill_gparams_from_args(gparams, args);
    fill_ie_params_from_args(ort_params, args["options"].as<std::string>());

    OrtConvTasNetInference interface{model, gparams, ort_params};

    std::cout << "Opening file " << args["file"].as<std::string>() << std::endl;
    // read audio file
    SF_INFO  info{};
    SNDFILE* in =
        sf_open(args["file"].as<std::string>().c_str(), SFM_READ, &info);
    SNDFILE* out = sf_open("output.flac", SFM_WRITE, &info);

    const int          BLOCK = 256;
    std::vector<float> buf(BLOCK);
    sf_count_t         n;
    // inference buffer by buffer
    std::cout << "Starting inferences..." << std::endl;

    std::vector<float> latencies;
    while ((n = sf_read_float(in, buf.data(), BLOCK)) > 0) {
        auto beg = std::chrono::high_resolution_clock::now();
        interface.run(buf.data(), BLOCK);
        auto end = std::chrono::high_resolution_clock::now();

        latencies.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)
                .count());
        sf_write_float(out, buf.data(), n);
    }

    npy::npy_data<float> latencies_npy;
    latencies_npy.data  = latencies;
    latencies_npy.shape = {latencies.size()};

    const std::string path{"convtasnet_latencies.npy"};
    npy::write_npy(path, latencies_npy);

    sf_close(in);
    sf_close(out);
    std::cout << "Job done" << std::endl;
}