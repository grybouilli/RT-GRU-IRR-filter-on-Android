cmake_minimum_required(VERSION 3.24)  
project(filtered LANGUAGES C CXX)  
set(CMAKE_CXX_STANDARD 20)            

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
set(SOURCE_FILES src/main.cpp src/acquire_audio_stream.cpp)
set(GET_OPT_MODEL_SRC_FILES src/get_optimized_model.cpp)
set(OFFLINE_INFERENCE_SRC_FILES src/offline_inferences.cpp)
set(GET_OPT_MODEL_BIN get_opt_model)
set(OFFLINE_INFERENCE offline_inference)

option(ONNX_USE_QNN "Will fetch a version of Onnx Runtime that only supports QNN and CPU" OFF)
option(APP_AS_APK "Will build app as a complete Android APK app" OFF)

set(ANIRA_WITH_TFLITE OFF)
set(ANIRA_WITH_LIBTORCH OFF)
set(LIBONNXRUNTIME_VERSION 1.24.1)
add_subdirectory(modules/anira)

include(cmake/liboboe.cmake)
include(cmake/DynamicONNXRuntime.cmake)
include(cmake/cxxopts.cmake)
include(cmake/magic_enum.cmake)
include(cmake/nlohmann_json.cmake)
include(cmake/boost_pfr.cmake)
include(cmake/FetchLibsamplerate.cmake)


if(NOT APP_AS_APK)
    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
    add_executable(${OFFLINE_INFERENCE} ${OFFLINE_INFERENCE_SRC_FILES})
else()
    add_compile_definitions(APP_AS_APK)
    add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} PRIVATE log)
endif()

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE oboe
    PRIVATE log
    PRIVATE cxxopts::cxxopts
    PRIVATE magic_enum::magic_enum
    PRIVATE anira
    PRIVATE nlohmann_json::nlohmann_json
    PRIVATE Boost::pfr
)

target_link_options(${PROJECT_NAME} PRIVATE -Wl,--allow-shlib-undefined)

target_include_directories(
    ${OFFLINE_INFERENCE}
    PRIVATE ${ONXX_RT_SRC_DIR}/headers
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_BINARY_DIR}/_deps/onnxruntime-src/headers
)

target_link_libraries(
    ${OFFLINE_INFERENCE}
    PRIVATE log
    PRIVATE cxxopts::cxxopts
    PRIVATE magic_enum::magic_enum
    PRIVATE nlohmann_json::nlohmann_json
    PRIVATE Boost::pfr
    PRIVATE ${CMAKE_SOURCE_DIR}/modules/libonnxruntime.so
    PRIVATE SampleRate::samplerate
    PRIVATE sndfile 
    PRIVATE fftw3
)
target_link_options(${OFFLINE_INFERENCE} PRIVATE -Wl,--allow-shlib-undefined)
