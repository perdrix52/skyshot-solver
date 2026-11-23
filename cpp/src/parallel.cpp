#include "skyshot/parallel.hpp"

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace skyshot {

namespace {
thread_local bool in_worker = false;

struct Pool {
  std::vector<std::thread> threads;
  std::mutex m;
  std::condition_variable cv;
  std::condition_variable done_cv;
  const std::function<void(size_t)>* fn = nullptr;
  size_t n = 0;
  std::atomic<size_t> next{0};
  size_t done = 0;
  uint64_t generation = 0;
  bool stop = false;
  std::mutex job_mutex;  // serialises concurrent parallel_for callers

  explicit Pool(size_t workers) {
    for (size_t i = 0; i < workers; ++i) threads.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> g(m);
      stop = true;
    }
    cv.notify_all();
    for (auto& t : threads) t.join();
  }
  void work() {
    for (;;) {
      size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) break;
      (*fn)(i);
    }
  }
  void loop() {
    in_worker = true;
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lk(m);
    for (;;) {
      cv.wait(lk, [&] { return stop || generation != seen; });
      if (stop) return;
      seen = generation;
      lk.unlock();
      work();
      lk.lock();
      if (++done == threads.size()) done_cv.notify_all();
    }
  }
  void run(size_t count, const std::function<void(size_t)>& f) {
    std::lock_guard<std::mutex> jg(job_mutex);
    {
      std::lock_guard<std::mutex> g(m);
      fn = &f;
      n = count;
      next.store(0);
      done = 0;
      ++generation;
    }
    cv.notify_all();
    bool was = in_worker;
    in_worker = true;
    work();
    in_worker = was;
    std::unique_lock<std::mutex> lk(m);
    done_cv.wait(lk, [&] { return done == threads.size(); });
  }
};

Pool* pool() {
  static Pool* p = [] {
    size_t n = thread_count();
    return n > 1 ? new Pool(n - 1) : nullptr;  // intentionally leaked: workers outlive static destruction order
  }();
  return p;
}
}  // namespace

size_t thread_count() {
  static size_t n = [] {
    if (const char* e = std::getenv("SKYSHOT_THREADS")) {
      long v = std::strtol(e, nullptr, 10);
      return size_t(v < 1 ? 1 : v);
    }
    unsigned h = std::thread::hardware_concurrency();
    return size_t(h ? h : 1);
  }();
  return n;
}

void parallel_for_impl(size_t n, const std::function<void(size_t)>& fn) {
  Pool* p = pool();
  if (!p || in_worker) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  p->run(n, fn);
}

}  // namespace skyshot
