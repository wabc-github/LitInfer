#include "profiler.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace llaisys {

thread_local std::string g_profile_phase;
thread_local int g_profile_depth = 0;

std::string read_environment(const char *key) {
#if defined(_MSC_VER)
    char *buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, key) != 0 || !buf) return {};
    std::string value(buf);
    std::free(buf);
    return value;
#else
    const char *value = std::getenv(key);
    return value ? std::string(value) : std::string();
#endif
}

namespace {

struct ProfilerState {
    std::mutex mutex;
    std::map<std::string, double> total_ms;
    std::map<std::string, int> calls;
};

ProfilerState &state() {
    // Deliberately leak the state. The atexit report must remain valid even
    // when the shared library's static destruction order is implementation
    // dependent on Windows.
    static ProfilerState *value = new ProfilerState();
    return *value;
}

} // namespace

bool Profiler::enabled() {
    static const bool on = [] {
        const std::string value = read_environment("LLAISYS_PROFILE");
        return value == "1" || value == "true" || value == "TRUE" ||
               value == "on" || value == "ON";
    }();
    return on;
}

void Profiler::record(const std::string &name, double seconds) {
    if (!enabled()) return;
    const std::string key = g_profile_phase.empty()
        ? name
        : g_profile_phase + "/" + name;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.total_ms[key] += seconds * 1000.0;
    ++s.calls[key];
}

void Profiler::reset() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.total_ms.clear();
    s.calls.clear();
}

void Profiler::report(std::ostream &os) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    double grand = 0.0;
    for (const auto &entry : s.total_ms) grand += entry.second;

    os << "\n================ LLAISYS PROFILE ================\n";
    if (grand <= 0.0) {
        os << "(no profiled operations)\n";
        os << "================================================\n";
        return;
    }

    std::vector<std::pair<std::string, double>> rows(
        s.total_ms.begin(), s.total_ms.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    os << std::fixed << std::setprecision(3);
    os << "  total op time: " << grand << " ms\n";
    os << "------------------------------------------------\n";
    os << "  op                 calls     time(ms)    pct(%)\n";
    for (const auto &row : rows) {
        const auto it = s.calls.find(row.first);
        const int calls = it == s.calls.end() ? 0 : it->second;
        const double pct = 100.0 * row.second / grand;
        os << "  " << std::left << std::setw(24) << row.first
           << std::right << std::setw(8) << calls
           << std::setw(12) << row.second
           << std::setw(9) << std::setprecision(2) << pct
           << std::setprecision(3) << "\n";
    }
    os << "================================================\n";
}

Profiler &profiler() {
    static Profiler *value = new Profiler();
    return *value;
}

AutoReport::AutoReport() {
    std::atexit(&AutoReport::report_at_exit);
}

void AutoReport::report_at_exit() {
    if (Profiler::enabled()) profiler().report(std::cerr);
}

AutoReport &auto_report() {
    static AutoReport *guard = new AutoReport();
    return *guard;
}

} // namespace llaisys
