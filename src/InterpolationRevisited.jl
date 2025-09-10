module InterpolationRevisited

export BSplinePoles, to_interpolation_coefficients!

using Neutrals # for fancy 𝟘 and 𝟙 neutral numbers

include("types.jl")
include("poles.jl")

end
