include(cmake/FetchLibsndfile.cmake)
include(FetchContent)

FetchContent_Declare(
    libsamplerate
    GIT_REPOSITORY https://github.com/libsndfile/libsamplerate
    GIT_TAG 0.2.2
)

FetchContent_MakeAvailable(libsamplerate)
