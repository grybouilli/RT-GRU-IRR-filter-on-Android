#include <App.hpp>
#include <ModelTypes.hpp>
#include <parsing_utils.hpp>

std::atomic<bool> run = true;

void sigint_handler(int arg) { run = false; }

template <typename IEParams>
int runApp(const cxxopts::ParseResult& args, GeneralInferenceParams& gparams) {
    IEParams params;
    fill_ie_params_from_args(params, args["options"].as<std::string>());

    if (args["model_type"].as<std::string>() ==
        magic_enum::enum_name(ModelType::GRU)) {
        static constexpr int32_t batch_size             = 1;
        static constexpr int32_t algo_audio_buffer_size = 256;
        static constexpr int32_t input_size             = 2;
        static constexpr int32_t hidden_size            = 128;
        static constexpr int32_t num_layers             = 2;
        IIRGRUInfo<batch_size,
                   algo_audio_buffer_size,
                   input_size,
                   hidden_size,
                   num_layers>
            gru;
        App app(gru, args, gparams, params, run);
        app.run();
    }

    return EXIT_SUCCESS;
}

int main_body(int argc, char** argv) {
    auto options = get_options();
    auto args    = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    GeneralInferenceParams gparams;
    fill_gparams_from_args(gparams, args);

    std::cout << std::format("normed cut off freq = {}", gparams.Fc_normed)
              << std::endl;
    if (gparams.chosen_engine == SupportedInferenceEngines::Ort) {
        return runApp<OrtParams>(args, gparams);
    }

    if (gparams.chosen_engine == SupportedInferenceEngines::Anira) {
        return runApp<AniraParams>(args, gparams);
    }
    return 0;
}

#ifndef APP_AS_APK
#pragma message("Compiling for native run")
int main(int argc, char** argv) {
    signal(SIGINT, sigint_handler);
    return main_body(argc, argv);
}
#else
#include <android/log.h>
#include <jni.h>
#include <pthread.h>
#include <unistd.h>

#define TAG "WrapperApp"

static void* stdoutToLogcat(void* arg) {
    int     fd = (int)(intptr_t)arg;
    char    buf[1024];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        __android_log_print(ANDROID_LOG_INFO, TAG, "%s", buf);
    }
    return nullptr;
}

static void redirectStdoutToLogcat() {
    int pipefd[2];
    pipe(pipefd);

    // Replace stdout and stderr with the write end of the pipe
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);

    // Spawn a thread to read the read end and forward to logcat
    pthread_t thread;
    pthread_create(&thread,
                   nullptr,
                   stdoutToLogcat,
                   (void*)(intptr_t)pipefd[0]);
    pthread_detach(thread);
}

#pragma message("Compiling for embedding in APK app")
std::vector<std::string> parseArgs(const std::string& input) {
    std::vector<std::string> tokens;
    std::string              current;
    char                     quoteChar = 0;
    bool                     inQuote   = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (std::isspace(c) && !current.empty()) {
            tokens.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);  // last token
    }

    return tokens;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_wrapperapp_MainActivity_runFiltered(JNIEnv* env,
                                                     jobject,
                                                     jstring jargs) {
    redirectStdoutToLogcat();  // call once before anything else
    const char* argsStr = env->GetStringUTFChars(jargs, nullptr);
    std::string argsStdStr(argsStr);
    env->ReleaseStringUTFChars(jargs, argsStr);

    std::vector<std::string> tokens = parseArgs(argsStdStr);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("filtered"));  // argv[0]
    for (auto& t : tokens) {
        argv.push_back(t.data());
    }
    int argc = argv.size();

    __android_log_print(ANDROID_LOG_INFO,
                        TAG,
                        "Calling main_body() with argc=%d",
                        argc);
    for (int i = 0; i < argc; i++) {
        __android_log_print(ANDROID_LOG_INFO,
                            TAG,
                            "  argv[%d] = %s",
                            i,
                            argv[i]);
    }

    main_body(argc, argv.data());
}
#endif