// `torch <command>`: the engine's command-line tool.
#include <cstdio>
#include <cstring>

#include "torch/version.h"

namespace torch {
int inspect(int argc, char** argv);  // tools/inspect.cpp
#if defined(TORCH_ENGINE)
int run(int argc, char** argv);  // tools/run.cpp
#endif
}  // namespace torch

namespace {

constexpr const char* kUsage =
    "usage: torch <command> [options]\n"
    "\n"
    "commands:\n"
    "  inspect     print a GGUF file's split, metadata, formats and tensors\n"
#if defined(TORCH_ENGINE)
    "  run         feed token ids through the model and decode greedily\n"
#endif
    "  --version   print the version and the git commit it was built from\n"
    "\n"
    "Run `torch <command> --help` for the options of a command.\n";

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "inspect") == 0)
        return torch::inspect(argc - 2, argv + 2);
#if defined(TORCH_ENGINE)
    if (argc >= 2 && std::strcmp(argv[1], "run") == 0)
        return torch::run(argc - 2, argv + 2);
#endif
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("torch %s (%s)\n", TORCH_VERSION, TORCH_GIT_SHA);
        return 0;
    }
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    std::fputs(kUsage, stderr);
    return 2;
}
