target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if is_plat("windows") then
        add_cxflags("/openmp", {tools = {"cl"}})
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if is_plat("windows") then
        -- 通用算子只依赖 OpenMP。AVX2 内核单独编译，运行时分派。
        add_cxflags("/openmp", {tools = {"cl"}})
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_files("../src/ops/add/cpu/*.cpp")
    add_files("../src/ops/linear/cpu/linear_cpu.cpp")

    -- INT8 weight-only 量化内核（通用 OpenMP + AVX2 向量化）
    add_files("../src/ops/quantize/cpu/quantize_cpu.cpp")
    if is_plat("windows") then
        add_files("../src/ops/quantize/cpu/quantize_cpu_avx.cpp", {
            cxflags = "/arch:AVX2",
            defines = "LLAISYS_COMPILE_AVX_QUANT"
        })
    else
        add_files("../src/ops/quantize/cpu/quantize_cpu_avx.cpp", {
            cxflags = {"-mavx2", "-mfma", "-mf16c"},
            defines = "LLAISYS_COMPILE_AVX_QUANT"
        })
    end

    if is_plat("windows") then
        add_files("../src/ops/linear/cpu/linear_cpu_avx.cpp", {
            cxflags = "/arch:AVX2",
            defines = "LLAISYS_COMPILE_AVX_LINEAR"
        })
    else
        add_files("../src/ops/linear/cpu/linear_cpu_avx.cpp", {
            cxflags = {"-mavx2", "-mfma", "-mf16c"},
            defines = "LLAISYS_COMPILE_AVX_LINEAR"
        })
    end

    on_install(function (target) end)
target_end()
