// Forced photometry at given positions (the evidence layer of the annotation atlas). Port of src/measure.rs.
#pragma once
#include <functional>
#include <memory>

#include "image_io.hpp"
#include "json.hpp"
#include "types.hpp"

namespace skyshot::measure {

using image_io::GrayImage;
using image_io::SourceInfo;

inline constexpr double R_AP = 1.0, R_IN = 2.5, R_OUT = 4.0;
inline constexpr double R_AP_DOG = 1.2, R_IN_DOG = 4.0, R_OUT_DOG = 6.0;
inline constexpr double SNR_RECORDED = 6.0, SNR_CONFIRM = 3.0, OFFSET_MAX = 0.75, SNR_TRACE = 3.0;
inline constexpr double SAT_FRACTION = 0.95, BRIGHT_LEVEL = 0.85, BRIGHT_CLIPPED = 0.1, MIN_ANNULUS = 0.5;
inline constexpr double FLAT_LSB = 0.5, FLAT_SKY = 0.05, BLEND_RADIUS = 1.5, BLEND_DEPTH = 1.0;
inline constexpr double SELF_ARCSEC = 1.5, SELF_FWHM = 0.25, ENV_FAINT = 0.5, ENV_BRIGHT = 5.0;
inline constexpr size_t N_CAL = 2000, CAL_MIN_ENV = 40, CAL_MIN = 10;
inline constexpr double CAL_EMPTY = 2.0;
inline constexpr double ZP_MIN_SNR = 20.0, LINEAR_SLOPE_TOL = 0.1, SNR_LIMIT = 5.0, SUSPECT_SNR = 2.0;
inline constexpr double MIN_FWHM = 0.5, MAX_FWHM = 64.0;
inline constexpr size_t MAX_TARGETS = 200000, MAX_REFS = 500000;

namespace flags {
inline constexpr uint32_t SATURATED = star_flags::SATURATED, BLENDED = star_flags::BLENDED, NEAR_EDGE = star_flags::NEAR_EDGE;
inline constexpr uint32_t NONLINEAR = 1u << 16, NO_DATA = 1u << 17, BRIGHT_BG = 1u << 18, OFFSET = 1u << 19, UNCONFIRMED = 1u << 20,
                          NO_DOG = 1u << 21, UNCALIBRATED = 1u << 22, SUSPECT = 1u << 23;
inline constexpr uint32_t NOT_ASSESSABLE = SATURATED | BLENDED | NEAR_EDGE | NO_DATA | BRIGHT_BG | UNCALIBRATED;
}  // namespace flags

enum class Status { Recorded, Trace, Below, NotAssessable };
const char* status_name(Status s);
enum class Env { Sky = 0, Faint = 1, Bright = 2 };
const char* env_name(Env e);
enum class FwhmSource { Given, Fit, FitRelaxed };

// Error of the measurement layer: bad_request, no_frame, no_psf, content_conflict.
struct MeasureError : std::runtime_error {
  std::string code;
  MeasureError(std::string c, const std::string& m) : std::runtime_error(m + " (" + c + ")"), code(std::move(c)) {}
};

struct Content {
  std::vector<std::array<double, 4>> include, exclude;
  bool operator==(const Content& o) const { return include == o.include && exclude == o.exclude; }
  Json to_json() const;
  static Content from_json(const Json& j);
};
// Set pixels outside `c` to NaN; returns how many were masked.
size_t apply_content(GrayImage& img, const Content& c);

struct FrameOptions {
  std::optional<double> fwhm;
  std::optional<bool> linear;
  std::optional<Content> content;
  Json to_json() const;
  static FrameOptions from_json(const Json& j);
};

struct RefStar {
  double x, y, mag;
};
struct Target {
  double x, y;
  std::optional<double> mag;
};

struct ZeroPoint {
  size_t n = 0;
  double zp = 0, scatter = 0;
  std::optional<double> slope, slope_se, intercept;
  std::array<double, 2> mag_range{0, 0};
  bool linear = false;
  Json to_json() const;
};
std::optional<ZeroPoint> fit_zero_point(const std::vector<double>& cat, const std::vector<double>& inst);

struct Chi {
  std::optional<double> sky, faint, bright, all;
  std::optional<double> get(Env e) const;
  Json to_json() const;
};
template <class T>
struct PerEnv {
  T sky{}, faint{}, bright{};
};

struct ModelSummary {
  uint32_t width = 0, height = 0;
  double fwhm = 0;
  FwhmSource fwhm_source = FwhmSource::Given;
  size_t n_psf = 0, bin = 0;
  double lsb = 0, level_sky = 0, sigma_sky = 0, saturation = 0, sat_thr = 0;
  bool linear = false, photometric = false;
  std::optional<double> gain;
  Chi chi, chi_dog;
  PerEnv<size_t> n_cal;
  std::optional<ZeroPoint> zero_point;
  std::optional<double> m_lim;
  PerEnv<std::optional<double>> m5_empirical;
  PerEnv<std::vector<std::array<double, 2>>> snr_curve;
  size_t n_stars = 0, n_refs = 0, n_refs_measured = 0;
  double self_px = 0;
  Json to_json() const;
};

struct Measurement {
  double x = 0, y = 0;
  std::optional<double> flux, sigma, snr, snr_cal, snr_dog, dx, dy, bg;
  size_t n_pix = 0;
  std::optional<Env> env;
  Status status = Status::Below;
  uint32_t flags = 0;
  Json to_json() const;
};

class FrameModel;
struct FrameModelImpl;

class FrameModel {
 public:
  // Model a frame; throws MeasureError.
  static FrameModel create(const GrayImage& img, const SourceInfo& info, const std::vector<Star>& stars, const std::vector<RefStar>& refs,
                           std::optional<double> scale_arcsec, const FrameOptions& opts);
  ~FrameModel();
  FrameModel(FrameModel&&) noexcept;
  FrameModel& operator=(FrameModel&&) noexcept;
  double fwhm() const;
  bool photometric() const;
  ModelSummary summary() const;
  Measurement measure_one(const GrayImage& img, const Target& t) const;

 private:
  FrameModel();
  std::unique_ptr<FrameModelImpl> impl_;
};

// Forced photometry at `targets`, in input order.
std::vector<Measurement> measure(const GrayImage& img, const FrameModel& model, const std::vector<Target>& targets);

// ---- request / response ----
struct Point {
  std::optional<double> x, y, ra, dec, mag;
  Json to_json() const;
  static Point from_json(const Json& j);
};
struct Request {
  std::optional<Wcs> wcs;
  std::vector<Point> targets, refs;
  FrameOptions frame;
  static Request from_json(const Json& j);
};
inline constexpr const char* COLUMNS[14] = {"x", "y", "flux", "sigma", "snr", "snr_cal", "snr_dog", "dx", "dy", "bg", "n_pix", "env", "status", "flags"};
struct Counts {
  size_t recorded = 0, trace = 0, below = 0, not_assessable = 0;
};
struct Response {
  ModelSummary model;
  Counts counts;
  std::vector<Json> rows;  // one 14-element array per target
  Json to_json() const;
};

class Session {
 public:
  std::optional<Content> content;
  std::string model_key;
  std::unique_ptr<FrameModel> model;
};

// Serve a request on an open frame (content mask applied once, model built or reused, then measured).
Response run(GrayImage& img, const SourceInfo& info, const Request& req, Session& session,
             const std::function<std::vector<Star>(const GrayImage&)>& stars);

}  // namespace skyshot::measure
