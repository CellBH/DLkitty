using DLkitty
using Documenter

DocMeta.setdocmeta!(DLkitty, :DocTestSetup, :(using DLkitty); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [DLkitty],
    authors = "",
    repo = "https://github.com/CellBH/DLkitty.jl/blob/{commit}{path}#{line}",
    sitename = "DLkitty.jl",
    format = Documenter.HTML(; canonical = "https://CellBH.github.io/DLkitty.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/CellBH/DLkitty.jl")
