#pragma once

#include <chrono>
#include <ostream>
#include <string>

namespace llaisys {

// Platform-safe environment lookup. The implementation is inline because it
// has no state; profiler aggregation itself lives in profiler.cpp so a DLL has
// exactly one counter table across all operator translation units.
std::string read_environment(const char *key);

extern thread_local std::string g_profile_phase;
extern thread_local int g_profile_depth;

class Profiler {
public:
    using clock = std::chrono::high_resolution_clock;

    static bool enabled();
    void record(const std::string &name, double seconds);
    void reset();
    void report(std::ostream &os);
};

Profiler &profiler();

struct ScopedProfilePhase {
    std::string previous;

    explicit ScopedProfilePhase(const char *phase)
        : previous(g_profile_phase) {
        g_profile_phase = phase ? phase : "";
    }

    ~ScopedProfilePhase() {
        g_profile_phase = previous;
    }
};

struct AutoReport {
    AutoReport();
    static void report_at_exit();
};

AutoReport &auto_report();

struct ScopedProfile {
    std::string name;
    Profiler::clock::time_point t0;
    bool counted = false;

    explicit ScopedProfile(std::string n)
        : name(std::move(n)), t0(Profiler::clock::now()) {
        auto_report();
        if (Profiler::enabled() && g_profile_depth == 0) counted = true;
        ++g_profile_depth;
    }

    ~ScopedProfile() {
        --g_profile_depth;
        if (counted) {
            const auto dt = std::chrono::duration<double>(
                Profiler::clock::now() - t0).count();
            profiler().record(name, dt);
        }
    }
};

} // namespace llaisys

#define PROFILE(name) ::llaisys::ScopedProfile _scoped_profile(name)
