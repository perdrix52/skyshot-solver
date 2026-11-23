// Minimal data-parallel helpers standing in for rayon.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

namespace skyshot {

// Number of worker threads used by parallel_for (SKYSHOT_THREADS overrides; 0/1 = serial).
size_t thread_count();

// Runs fn(i) for i in [0, n); iterations are independent. Nested calls run serially.
void parallel_for_impl(size_t n, const std::function<void(size_t)>& fn);

template <class F>
void parallel_for(size_t n, F&& fn) {
  if (n <= 1 || thread_count() <= 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  parallel_for_impl(n, std::function<void(size_t)>(std::forward<F>(fn)));
}

// Runs fn(begin, end) over contiguous blocks of at most `chunk` indices.
template <class F>
void parallel_chunks(size_t n, size_t chunk, F&& fn) {
  if (chunk == 0) chunk = 1;
  size_t blocks = (n + chunk - 1) / chunk;
  parallel_for(blocks, [&](size_t b) { fn(b * chunk, std::min(n, (b + 1) * chunk)); });
}

// Parallel map into a pre-sized output vector.
template <class T, class F>
std::vector<T> parallel_map(size_t n, F&& fn) {
  std::vector<T> out(n);
  parallel_for(n, [&](size_t i) { out[i] = fn(i); });
  return out;
}

}  // namespace skyshot
