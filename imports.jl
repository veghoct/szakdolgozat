import Pkg
for pkg in ["DataFrames", "Plots"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using Random
using Statistics
using DataFrames
using Plots