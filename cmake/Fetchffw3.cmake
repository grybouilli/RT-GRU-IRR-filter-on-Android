include(FetchContent)

FetchContent_Declare(
    ffw3
    URL      https://fftw.org/fftw-3.3.11.tar.gz
)

# Tell fftw3's own CMake to build as static and enable what libsndfile needs
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(FFTW_BUILD_TESTS OFF CACHE BOOL "" FORCE)

message(STATUS "[FetchMyLibrary] Fetching fftw 3.3.11 ...")
FetchContent_MakeAvailable(ffw3)
message(STATUS "[FetchMyLibrary] fftw3 3.3.11 is ready.")