// Generate a tiny artificial catalogue and star list for a full offline CLI demonstration.
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "support.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fputs("usage: synthetic_demo NEW_DIRECTORY\n", stderr);
    return 2;
  }
  try {
    std::filesystem::path root = std::filesystem::u8path(argv[1]);
    if (!std::filesystem::create_directory(root)) throw skyshot::Error("Use a new directory with an existing parent");
    std::optional<std::filesystem::path> coder;
    if (argc > 2) coder = std::filesystem::u8path(argv[2]);
    auto [field, expected] = support::synthetic_catalog(root / "index", coder);
    std::ofstream(root / "stars.json", std::ios::binary) << field.to_json().dump(true);
    std::ofstream(root / "expected-wcs.json", std::ios::binary) << expected.to_json().dump(true);
    std::printf("Created artificial index and stars in %s\n", root.string().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "synthetic_demo: %s\n", e.what());
    return 2;
  }
}
