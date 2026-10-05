"""Build only RouteOpt's pricing kernel; no external installations or solver run."""
from pathlib import Path
import os
import shutil
import subprocess


def build():
    here = Path(__file__).resolve().parent
    upstream = here.parents[3] / "RouteOpt-main"
    packages = upstream / "packages"
    output = here / "build"
    # Invalidate before refreshing overlay files; a failed build cannot leave
    # an apparently valid receipt for an older binary. No runtime auto-build.
    (output / "verified_build.json").unlink(missing_ok=True)
    overlay = output / "include"
    overlay.mkdir(parents=True, exist_ok=True)
    pricing = packages / "application/cvrp/src/pricing"
    for header in (pricing / "include").glob("*.hpp"):
        shutil.copy2(header, overlay / header.name)
    changes = {
        "pricing_macro.hpp": [("MAX_ROUTE_PRICING = 6400000", "MAX_ROUTE_PRICING = 64000")],
        "write_columns_from_pricing.hpp": [
            ("            delete[] all_label;\n            label_assign *= 2;",
             '            if (label_assign >= 262144) throw std::runtime_error("adapter label limit");\n'
             "            delete[] all_label;\n            label_assign *= 2;")],
    }
    rank_macro = packages / "rank1_cuts/common/include/rank1_macro.hpp"
    shutil.copy2(rank_macro, overlay / rank_macro.name)
    changes[rank_macro.name] = [
        ("MAX_NUM_R1CS_IN_PRICING = 2048", "MAX_NUM_R1CS_IN_PRICING = 1"),
        ("MAX_POSSIBLE_NUM_R1CS_FOR_VERTEX = 128", "MAX_POSSIBLE_NUM_R1CS_FOR_VERTEX = 1"),
    ]
    for name, replacements in changes.items():
        file = overlay / name
        content = file.read_text()
        for old, new in replacements:
            if content.count(old) != 1:
                raise RuntimeError(f"upstream overlay mismatch: {name}: {old}")
            content = content.replace(old, new)
        file.write_text(content)
    includes = [overlay, packages / "common/config", packages / "external/eigen",
                packages / "external/boost"]
    for directory in [packages / "application/cvrp/src", packages / "rank1_cuts",
                      packages / "rounded_cap_cuts", packages / "common"]:
        includes.extend(sorted(directory.rglob("include")))
    gurobi = os.environ.get("GUROBI_INCLUDE_DIR")
    if gurobi is None:
        candidates = sorted(Path("/Library").glob("gurobi*/macos_universal2/include/gurobi_c.h"))
        if not candidates:
            raise RuntimeError("Set GUROBI_INCLUDE_DIR to the existing Gurobi C header directory")
        gurobi = str(candidates[-1].parent)
    includes.append(Path(gurobi))
    sources = [here / "pricing_cli.cpp", *sorted((pricing / "src").glob("*.cpp")),
               packages / "rank1_cuts/common/src/rank1cuts.cpp",
               *sorted((packages / "rank1_cuts/chg_rc_getter/src").glob("*.cpp")),
               packages / "rounded_cap_cuts/chg_rc_getter/src/get_chg_rcc_rc.cpp"]
    command = [os.environ.get("CXX", "clang++"), "-std=c++20", "-O3", "-DNDEBUG",
               "-DLABEL_ASSIGN_OVERRIDE=2048", "-fno-fast-math"]
    for directory in includes:
        command.extend(["-I", str(directory)])
    command.extend(map(str, sources))
    command.extend(["-o", str(output / "routeopt_pricing")])
    from verified_pricing import compile_verified
    compile_verified(command)
    print(output / "routeopt_pricing")


if __name__ == "__main__":
    build()
