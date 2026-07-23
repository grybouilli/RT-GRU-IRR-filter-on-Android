#include <sndfile.h>

#include <ModelInferenceMethods/ConvTasNetMethods/Ort/OrtConvTasNetInference.hpp>
#include <ModelInferenceMethods/GeneralInferenceParams.hpp>
#include <ModelInferenceMethods/IEParams.hpp>
#include <ModelInferenceMethods/OrtUtils/OrtSessionHandler.hpp>
#include <ModelInferenceMethods/OrtUtils/OrtTensorBuffer.hpp>
#include <nlohmann/json.hpp>
#include <npy.hpp>
#include <parsing_utils.hpp>
#include <random>

using json = nlohmann::json;

constexpr size_t        upto           = 20;
static constexpr size_t in_shape_size  = 2;
static constexpr size_t out_shape_size = 3;
#define IN_SHAPE  {1, 24000}
#define OUT_SHAPE {1, 512, 3002}

int main(int argc, char** argv) {
    GeneralInferenceParams            gparams;
    OrtParams                         ort_params;
    ConvTasNetInfo<1, 24000, 2, 8000> model;

    auto options = get_options();
    options.add_options()("ctx_dest",
                          "Directory to which to save the compiled model",
                          cxxopts::value<std::string>()->default_value(""));
    auto args = options.parse(argc, argv);

    fill_gparams_from_args(gparams, args);
    fill_ie_params_from_args(ort_params, args["options"].as<std::string>());

    // Find loaded model filename and folder
    const std::string current_model_folder =
        gparams.model_filename.substr(0,
                                      gparams.model_filename.find_last_of("/"));
    const std::string current_model_basename = gparams.model_filename.substr(
        gparams.model_filename.find_last_of("/") + 1);
    std::string compiled_model_filename = current_model_basename;
    std::string ep_opt_json_file        = "";
    // add ep options if provided by user
    if (const auto ep_options_json =
            args["ort_json_ep_options"].as<std::string>();
        ep_options_json != "") {
        std::ifstream f(ep_options_json);
        json          ep_options = json::parse(f,
                                               /* callback */ nullptr,
                                               /* allow exceptions */ true,
                                               /* ignore_comments */ true);

        ort_params.EP_options = ep_options;

        const std::string base_filename =
            ep_options_json.substr(ep_options_json.find_last_of("/") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        const std::string file_without_extension = base_filename.substr(0, p);

        compiled_model_filename =
            file_without_extension + "_" + current_model_basename;
        ep_opt_json_file = file_without_extension;
    }

    if (const auto dest_ctx_folder = args["ctx_dest"].as<std::string>();
        dest_ctx_folder != "") {
        compiled_model_filename =
            dest_ctx_folder + "/" + compiled_model_filename;
    } else {
        compiled_model_filename =
            current_model_folder + compiled_model_filename;
    }

    if (ort_params.EP_options.contains("profiling_level")) {
        ort_params.EP_options["profiling_file_path"] =
            "logs/" + ep_opt_json_file + "_" + current_model_basename + ".csv";
        std::cout << "Profiling data will be put at "
                  << ort_params.EP_options["profiling_file_path"] << std::endl;
    }

    if (const auto config_entries_json =
            args["ort_json_config_entries"].as<std::string>();
        config_entries_json != "") {
        std::ifstream f(config_entries_json);
        json          config_entries = json::parse(f);

        ort_params.config_entries = config_entries;
    } else {
        // add entries to export .ctx model
        ort_params.config_entries[kOrtSessionOptionEpContextEnable]    = "1";
        ort_params.config_entries[kOrtSessionOptionEpContextEmbedMode] = "1";
        ort_params.config_entries[kOrtSessionOptionEpContextFilePath] =
            compiled_model_filename;
    }

    OrtConvTasNetInference interface{model, gparams, ort_params};
    return 0;
}