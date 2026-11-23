// Minimal JSON value, parser and serde_json-compatible writer.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common.hpp"

namespace skyshot {

class Json {
 public:
  enum class Type { Null, Bool, Int, UInt, Double, String, Array, Object };
  using Object = std::vector<std::pair<std::string, Json>>;
  using Array = std::vector<Json>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : type_(Type::Bool), b_(b) {}
  Json(int v) : type_(Type::Int), i_(v) {}
  Json(int64_t v) : type_(Type::Int), i_(v) {}
  Json(uint32_t v) : type_(Type::UInt), u_(v) {}
  Json(uint64_t v) : type_(Type::UInt), u_(v) {}
  Json(double v) : type_(Type::Double), d_(v) {}
  Json(float v) : type_(Type::Double), d_(double(v)) {}
  Json(const char* s) : type_(Type::String), s_(s) {}
  Json(std::string s) : type_(Type::String), s_(std::move(s)) {}
  Json(Array a) : type_(Type::Array), a_(std::move(a)) {}

  static Json object() {
    Json j;
    j.type_ = Type::Object;
    return j;
  }
  static Json array() {
    Json j;
    j.type_ = Type::Array;
    return j;
  }
  template <class T>
  static Json from_vec(const std::vector<T>& v) {
    Json j = array();
    for (const auto& x : v) j.push(Json(x));
    return j;
  }
  template <size_t N>
  static Json from_arr(const std::array<double, N>& v) {
    Json j = array();
    for (double x : v) j.push(Json(x));
    return j;
  }
  static Json opt(const std::optional<double>& v) { return v ? Json(*v) : Json(); }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Int || type_ == Type::UInt || type_ == Type::Double; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const { return b_; }
  double as_f64() const;                       // any number
  std::optional<uint64_t> as_u64() const;      // non-negative integers only
  std::optional<int64_t> as_i64() const;
  const std::string& as_str() const { return s_; }
  const Array& items() const { return a_; }
  Array& items() { return a_; }
  const Object& members() const { return o_; }

  Json& push(Json v) {
    a_.push_back(std::move(v));
    return *this;
  }
  Json& set(const std::string& k, Json v) {
    for (auto& kv : o_)
      if (kv.first == k) {
        kv.second = std::move(v);
        return *this;
      }
    o_.emplace_back(k, std::move(v));
    return *this;
  }
  const Json* get(std::string_view k) const {
    for (auto& kv : o_)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }

  std::string dump(bool pretty = false) const;
  void dump_to(std::string& out, bool pretty, int indent = 0) const;

 private:
  Type type_ = Type::Null;
  bool b_ = false;
  int64_t i_ = 0;
  uint64_t u_ = 0;
  double d_ = 0.0;
  std::string s_;
  Array a_;
  Object o_;
};

// Throws skyshot::Error on malformed input.
Json parse_json(std::string_view text);

// Shortest round-trip double formatting identical to serde_json (ryu); non-finite -> null elsewhere.
std::string format_f64(double v);
std::string format_f32(float v);

}  // namespace skyshot
