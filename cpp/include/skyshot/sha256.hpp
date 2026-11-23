// Self-contained SHA-256.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace skyshot {

class Sha256 {
 public:
  Sha256();
  void update(const void* data, size_t len);
  void update(std::string_view s) { update(s.data(), s.size()); }
  // Returns the 32-byte digest and resets nothing; call once.
  void finalize(uint8_t out[32]);
  std::string hex();  // lower-case hex digest

 private:
  void block(const uint8_t* p);
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t buflen_ = 0;
  uint64_t total_ = 0;
};

std::string sha256_hex(const void* data, size_t len);

}  // namespace skyshot
