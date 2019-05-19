using Documenter
using TensorPolynomialBases

makedocs(
    sitename = "TensorPolynomialBases.jl",
    format = Documenter.HTML(),
    modules = [TensorPolynomialBases],
    pages = ["Home" => "index.md","API"=>"pages/api.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

deploydocs(repo="github.com/gridap/TensorPolynomialBases.jl.git")
