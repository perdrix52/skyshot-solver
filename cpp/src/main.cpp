// Command-line interface. Port of src/main.rs (same commands, options, output and exit codes).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "skyshot/builder.hpp"
#include "skyshot/engine.hpp"
#include "skyshot/extract.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace skyshot;
namespace fs = std::filesystem;

namespace {

const char* VERSION = "0.1.0";

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

const char* MAIN_HELP =
    "Offline plate solving for developers \xE2\x80\x94 CLI and Rust library\n"
    "\n"
    "Usage: skyshot <COMMAND>\n"
    "\n"
    "Commands:\n"
    "  solve    Solve an image (FITS/PNG/JPEG/TIFF) or StarField JSON, writing JSON to stdout\n"
    "  batch    Solve several files with one open index; emit one JSON result per line\n"
    "  extract  Detect stars and emit a StarField JSON document, without an index\n"
    "  inspect  Open an index and print its manifest (reader checks, not a full checksum audit)\n"
    "  build    Build an index from a separately obtained catalogue. Always use a NEW output directory\n"
    "  help     Print this message or the help of the given subcommand(s)\n"
    "\n"
    "Options:\n"
    "  -h, --help     Print help\n"
    "  -V, --version  Print version\n";

const char* SOLVE_OPTIONS =
    "      --index <INDEX>\n"
    "      --mode <MODE>              [default: auto] [possible values: auto, standard, deep]\n"
    "      --deadline-ms <DEADLINE_MS>  [default: 2000]\n"
    "      --min-fov <MIN_FOV>\n"
    "      --max-fov <MAX_FOV>\n"
    "      --matches\n"
    "      --arc                      Allow an ARC projection for very wide frames\n"
    "      --panorama                 Opt in to STG and cylindrical panorama WCS models\n"
    "  -h, --help                     Print help\n";

void print_help(const std::string& cmd) {
  if (cmd == "solve") {
    std::fputs(
        "Solve an image (FITS/PNG/JPEG/TIFF) or StarField JSON, writing JSON to stdout\n\nUsage: skyshot solve [OPTIONS] --index <INDEX> <INPUT>\n\nArguments:\n  <INPUT>\n\nOptions:\n",
        stdout);
    std::fputs(SOLVE_OPTIONS, stdout);
    std::fputs("      --wcs <WCS>                Write a new FITS WCS header file on success (refuses existing files)\n", stdout);
  } else if (cmd == "batch") {
    std::fputs("Solve several files with one open index; emit one JSON result per line\n\nUsage: skyshot batch [OPTIONS] --index <INDEX> <INPUTS>...\n\nArguments:\n  <INPUTS>...\n\nOptions:\n", stdout);
    std::fputs(SOLVE_OPTIONS, stdout);
  } else if (cmd == "extract") {
    std::fputs("Detect stars and emit a StarField JSON document, without an index\n\nUsage: skyshot extract <INPUT>\n\nArguments:\n  <INPUT>\n\nOptions:\n  -h, --help  Print help\n", stdout);
  } else if (cmd == "inspect") {
    std::fputs("Open an index and print its manifest (reader checks, not a full checksum audit)\n\nUsage: skyshot inspect <INDEX>\n\nArguments:\n  <INDEX>\n\nOptions:\n  -h, --help  Print help\n", stdout);
  } else if (cmd == "build") {
    std::fputs(
        "Build an index from a separately obtained catalogue. Always use a NEW output directory\n\nUsage: skyshot build [OPTIONS] --d80 <D80> --out <OUT>\n\nOptions:\n"
        "      --d80 <D80>\n      --out <OUT>\n      --profile <PROFILE>  [default: full] [possible values: lite, standard, full, deep]\n"
        "      --coder <CODER>      Learned coder prefix (.f32 and .json); omitted means the handcrafted v1 key\n  -h, --help           Print help\n",
        stdout);
  } else {
    std::fputs(MAIN_HELP, stdout);
  }
}

struct Options {
  std::string index;
  bool has_index = false;
  SolveMode mode = SolveMode::Auto;
  double deadline_ms = 2000.0;
  std::optional<double> min_fov, max_fov;
  bool matches = false, arc = false, panorama = false;

  solve::SolveConfig config() const {
    solve::SolveConfig c;
    c.mode = mode;
    c.deadline_ms = deadline_ms;
    c.min_fov_deg = min_fov.value_or(c.min_fov_deg);
    c.max_fov_deg = max_fov.value_or(c.max_fov_deg);
    c.keep_matches = matches;
    c.allow_arc = arc;
    c.allow_panorama = panorama;
    return c;
  }
};

// Splits "--name=value" and returns the option name and an inline value.
struct Tok {
  std::string name;
  std::optional<std::string> value;
};
Tok split_opt(const std::string& a) {
  size_t eq = a.find('=');
  if (a.rfind("--", 0) == 0 && eq != std::string::npos) return {a.substr(0, eq), a.substr(eq + 1)};
  return {a, std::nullopt};
}

double parse_double_arg(const std::string& name, const std::string& v) {
  auto d = parse_f64(v);
  if (!d) throw UsageError("invalid value '" + v + "' for '" + name + " <" + name.substr(2) + ">': invalid float literal");
  return *d;
}

std::string take_value(const std::string& name, const std::optional<std::string>& inline_v, const std::vector<std::string>& args, size_t& i) {
  if (inline_v) return *inline_v;
  if (i + 1 >= args.size()) throw UsageError("a value is required for '" + name + "' but none was supplied");
  return args[++i];
}

// Parses the options shared by solve and batch. Positional arguments are collected.
void parse_solve_args(const std::vector<std::string>& args, size_t from, Options& o, std::vector<std::string>& positional,
                      std::optional<std::string>* wcs) {
  for (size_t i = from; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a.size() > 1 && a[0] == '-' && !(a.size() > 1 && (std::isdigit(static_cast<unsigned char>(a[1])) || a[1] == '.'))) {
      Tok t = split_opt(a);
      if (t.name == "--index") {
        o.index = take_value(t.name, t.value, args, i);
        o.has_index = true;
      } else if (t.name == "--mode") {
        std::string v = take_value(t.name, t.value, args, i);
        if (v == "auto") o.mode = SolveMode::Auto;
        else if (v == "standard") o.mode = SolveMode::Standard;
        else if (v == "deep") o.mode = SolveMode::Deep;
        else throw UsageError("invalid value '" + v + "' for '--mode <MODE>'\n  [possible values: auto, standard, deep]");
      } else if (t.name == "--deadline-ms") {
        o.deadline_ms = parse_double_arg(t.name, take_value(t.name, t.value, args, i));
      } else if (t.name == "--min-fov") {
        o.min_fov = parse_double_arg(t.name, take_value(t.name, t.value, args, i));
      } else if (t.name == "--max-fov") {
        o.max_fov = parse_double_arg(t.name, take_value(t.name, t.value, args, i));
      } else if (t.name == "--matches") {
        o.matches = true;
      } else if (t.name == "--arc") {
        o.arc = true;
      } else if (t.name == "--panorama") {
        o.panorama = true;
      } else if (t.name == "--wcs" && wcs) {
        *wcs = take_value(t.name, t.value, args, i);
      } else {
        throw UsageError("unexpected argument '" + a + "' found");
      }
    } else {
      positional.push_back(a);
    }
  }
  if (!o.has_index) throw UsageError("the following required arguments were not provided:\n  --index <INDEX>");
}

bool help_requested(const std::vector<std::string>& args, size_t from) {
  for (size_t i = from; i < args.size(); ++i)
    if (args[i] == "-h" || args[i] == "--help") return true;
  return false;
}

bool ends_with_json(const std::string& p) {
  std::string e = fs::path(p).extension().string();
  return e.size() == 5 && ascii_lower(e) == ".json";
}

SolveOutput solve_one(const Engine& engine, const std::string& path, const solve::SolveConfig& cfg) {
  if (ends_with_json(path)) {
    StarField field;
    try {
      field = StarField::from_json(parse_json(read_file_string(path)));
    } catch (const Error& e) {
      throw Error("Invalid StarField JSON: " + path + ": " + e.what());
    }
    return engine.solve_sources(field, cfg);
  }
  return engine.solve_path(fs::u8path(path), cfg);
}

void write_stdout(const std::string& s) {
  std::fwrite(s.data(), 1, s.size(), stdout);
}

int run(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::fputs(MAIN_HELP, stderr);
    return 2;
  }
  const std::string& cmd = args[0];
  if (cmd == "-h" || cmd == "--help") {
    print_help("");
    return 0;
  }
  if (cmd == "help") {
    print_help(args.size() > 1 ? args[1] : "");
    return 0;
  }
  if (cmd == "-V" || cmd == "--version") {
    std::printf("skyshot %s\n", VERSION);
    return 0;
  }
  int exit_code = 0;
  if (cmd == "solve") {
    if (help_requested(args, 1)) {
      print_help("solve");
      return 0;
    }
    Options o;
    std::vector<std::string> pos;
    std::optional<std::string> wcs;
    parse_solve_args(args, 1, o, pos, &wcs);
    if (pos.empty()) throw UsageError("the following required arguments were not provided:\n  <INPUT>");
    if (pos.size() > 1) throw UsageError("unexpected argument '" + pos[1] + "' found");
    if (wcs && fs::exists(fs::u8path(*wcs))) throw Error("WCS output already exists: " + *wcs);
    Engine engine = Engine::open(fs::u8path(o.index));
    SolveOutput result = solve_one(engine, pos[0], o.config());
    if (result.solution.status == "solved") {
      if (wcs) {
        if (!result.solution.wcs) throw Error("Solved result has no WCS");
        std::string header = result.solution.wcs->fits_header();
        fs::path wp = fs::u8path(*wcs);
        if (fs::exists(wp)) throw Error("File exists: " + *wcs);
        std::ofstream f(wp, std::ios::binary);
        if (!f) throw Error("create " + *wcs);
        f.write(header.data(), std::streamsize(header.size()));
      }
    } else {
      exit_code = 1;
    }
    write_stdout(result.to_json().dump(true) + "\n");
    return exit_code;
  }
  if (cmd == "batch") {
    if (help_requested(args, 1)) {
      print_help("batch");
      return 0;
    }
    Options o;
    std::vector<std::string> pos;
    parse_solve_args(args, 1, o, pos, nullptr);
    if (pos.empty()) throw UsageError("the following required arguments were not provided:\n  <INPUTS>...");
    Engine engine = Engine::open(fs::u8path(o.index));
    solve::SolveConfig cfg = o.config();
    for (const std::string& input : pos) {
      Json row = Json::object();
      try {
        SolveOutput result = solve_one(engine, input, cfg);
        if (result.solution.status != "solved") exit_code = std::max(exit_code, 1);
        row.set("input", input).set("result", result.to_json());
      } catch (const Error& e) {
        exit_code = 2;
        row = Json::object();
        row.set("input", input).set("error", std::string(e.what()));
      }
      write_stdout(row.dump(false) + "\n");
      std::fflush(stdout);
    }
    return exit_code;
  }
  if (cmd == "extract") {
    if (help_requested(args, 1)) {
      print_help("extract");
      return 0;
    }
    if (args.size() < 2) throw UsageError("the following required arguments were not provided:\n  <INPUT>");
    if (args.size() > 2) throw UsageError("unexpected argument '" + args[2] + "' found");
    auto loaded = image_io::load_with_info(fs::u8path(args[1]));
    StarField field = extract(loaded.image, ExtractOptions::for_source(loaded.info));
    write_stdout(field.to_json().dump(true) + "\n");
    return 0;
  }
  if (cmd == "inspect") {
    if (help_requested(args, 1)) {
      print_help("inspect");
      return 0;
    }
    if (args.size() < 2) throw UsageError("the following required arguments were not provided:\n  <INDEX>");
    if (args.size() > 2) throw UsageError("unexpected argument '" + args[2] + "' found");
    auto index = index::Index::open(fs::u8path(args[1]));
    write_stdout(index->manifest.to_json().dump(true) + "\n");
    return 0;
  }
  if (cmd == "build") {
    if (help_requested(args, 1)) {
      print_help("build");
      return 0;
    }
    std::optional<std::string> d80, out, coder;
    std::string profile = "full";
    for (size_t i = 1; i < args.size(); ++i) {
      Tok t = split_opt(args[i]);
      if (t.name == "--d80") d80 = take_value(t.name, t.value, args, i);
      else if (t.name == "--out") out = take_value(t.name, t.value, args, i);
      else if (t.name == "--profile") {
        profile = take_value(t.name, t.value, args, i);
        if (profile != "lite" && profile != "standard" && profile != "full" && profile != "deep")
          throw UsageError("invalid value '" + profile + "' for '--profile <PROFILE>'\n  [possible values: lite, standard, full, deep]");
      } else if (t.name == "--coder") coder = take_value(t.name, t.value, args, i);
      else throw UsageError("unexpected argument '" + args[i] + "' found");
    }
    if (!d80 || !out) throw UsageError("the following required arguments were not provided:\n  --d80 <D80>\n  --out <OUT>");
    std::error_code ec;
    fs::path outp = fs::u8path(*out);
    if (!fs::create_directory(outp, ec) || ec) throw Error("Use a new output directory with an existing parent");
    builder::BuildConfig cfg;
    cfg.d80 = fs::u8path(*d80);
    cfg.out = outp;
    cfg.r7_arcmin = profile == "deep" ? 0.485 : 1.94;
    cfg.max_pool_level = profile == "lite" ? 6 : (profile == "standard" ? 7 : 8);
    cfg.dump_levels = false;
    cfg.profile = profile;
    if (coder) cfg.coder = fs::u8path(*coder);
    auto manifest = builder::build(cfg);
    write_stdout(manifest.to_json().dump(true) + "\n");
    return 0;
  }
  throw UsageError("unrecognized subcommand '" + cmd + "'");
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  std::vector<std::string> args(argv + 1, argv + argc);
  try {
    return run(args);
  } catch (const UsageError& e) {
    std::fprintf(stderr, "error: %s\n\nFor more information, try '--help'.\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "skyshot: %s\n", e.what());
    return 2;
  }
}
