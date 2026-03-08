target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")

    add_rules("cuda")
    set_values("cuda.rdc", false)

    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart", "cublas")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
