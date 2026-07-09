import Pkg

for pkg in ["DataFrames", "CSV", "Plots", "Distances", "Clustering", "Distributions", "FastGaussQuadrature", "Optim"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using Random

using DataFrames

using Plots
using CSV

using Distances
using Clustering
using Statistics
using Optim
using Distributions
using FastGaussQuadrature
