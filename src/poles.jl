"""
    bspline_poles(deg, T=BigFloat; nbits=180)

Return the computed poles of the B-spline of degree `deg` with floating-point type `T`.

Keyword `nbits` is to specify the number of bits for the precision of computations. In the
original code, constants were given with 50 significant digits, that's slightly more than
166 bits.

# References

* P. Thévenaz, T. Blu, and M. Unser, "*Interpolation Revisited,*" IEEE Transactions on
  Medical Imaging, vol. **19**, no. 7, pp. 739-758, July 2000.

"""
function bspline_poles(deg::Integer, ::Type{T}; kwds...) where {T<:AbstractFloat}
    return map(T, bspline_poles(deg, BigFloat; kwds...))::Vector{T}
end
function bspline_poles(deg::Integer, ::Type{BigFloat} = BigFloat; nbits::Integer=180)
    setprecision(BigFloat, nbits) do
        if deg == 2
            return [sqrt(big"8.0") - big"3.0"]
        elseif deg == 3
            return [sqrt(big"3.0") - big"2.0"]
        elseif deg == 4
            return [sqrt(big"664.0" - sqrt(big"438976.0")) + sqrt(big"304.0") - big"19.0",
                    sqrt(big"664.0" + sqrt(big"438976.0")) - sqrt(big"304.0") - big"19.0"]
        elseif deg == 5
            return [
                (sqrt(big"135.0" / big"2.0" - sqrt(big"17745.0" / big"4.0"))
                 + sqrt(big"105.0" / big"4.0") - big"13.0" / big"2.0"),
                (sqrt(big"135.0" / big"2.0" + sqrt(big"17745.0" / big"4.0"))
                 - sqrt(big"105.0" / big"4.0") - big"13.0" / big"2.0")]
        elseif deg == 6
            return [big"-0.48829458930304475513011803888378906211227916123938",
                    big"-0.081679271076237512597937765737059080653379610398148",
                    big"-0.0014141518083258177510872439765585925278641690553467"]
        elseif deg == 7
            return [big"-0.53528043079643816554240378168164607183392315234269",
                    big"-0.12255461519232669051527226435935734360548654942730",
                    big"-0.0091486948096082769285930216516478534156925639545994"]
        elseif deg == 8
            return [big"-0.57468690924876543053013930412874542429066157804125",
                    big"-0.16303526929728093524055189686073705223476814550830",
                    big"-0.023632294694844850023403919296361320612665920854629",
                    big"-0.00015382131064169091173935253018402160762964054070043"]
        elseif deg == 9
            return [big"-0.60799738916862577900772082395428976943963471853991",
                    big"-0.20175052019315323879606468505597043468089886575747",
                    big"-0.043222608540481752133321142979429688265852380231497",
                    big"-0.0021213069031808184203048965578486234220548560988624"]
        else
            throw(ArgumentError("only B-spline of degrees 2 to 9 are supported, got degree $deg"))
        end
    end
end

# Accessor.
Base.values(x::BSplinePoles) = getfield(x, :values)

"""
    BSplinePoles{D,T=Float64}()

Return the computed poles for a B-spline of degree D. Optional type parameter `T` is the
floating-point type. Double precision poles are pre-computed at compilation time.

"""
BSplinePoles{D}() where {D} = BSplinePoles{D,Float64}()
BSplinePoles{D,Float32}() where {D} = BSplinePoles{D}(map(Float32, values(BSplinePoles{D}())))

# FIXME not type-stable
BSplinePoles{D,BigFloat}(; kwds...) where {D} =
    BSplinePoles{D}(bspline_poles(D, BigFloat; kwds...)...)

"""
    BSplinePoles{D,T}(vals...)

Build an object storing the poles `vals...` for a B-spline of degree D. Type parameter `T`
is the floating-point type, if not specified, it is inferred from the types of the poles.

"""
BSplinePoles{D}(vals::T...) where {D,T<:AbstractFloat} = BSplinePoles{D,T}(vals...)
BSplinePoles{D}(vals::Real...) where {D} = BSplinePoles{D}(promote(vals...)...)
BSplinePoles{D}(vals::Integer...) where {D} = BSplinePoles{D,Float64}(vals...)

# Poles may also be specified as a tuple.
BSplinePoles{D}(vals::Tuple{Vararg{Real}}) where {D} = BSplinePoles{D}(vals...)
BSplinePoles{D,T}(vals::Tuple{Vararg{Real}}) where {D,T<:AbstractFloat} =
    BSplinePoles{D,T}(vals...)

# Pre-compute B-spline poles in double precision. The number of bits of precision is specified
# to avoid that the values depend on the settings at compilation time.
for deg in 2:9
    @eval begin
        const $(Symbol("BSPLINE_POLES_", deg)) =
            $((bspline_poles(deg, Float64; nbits=60)...,))
        BSplinePoles{$deg,Float64}() =
            BSplinePoles{$deg}($(Symbol("BSPLINE_POLES_", deg)))
    end
end

index_bounds(A::AbstractArray) = (firstindex(A)::Int, lastindex(A)::Int)

"""
    to_interpolation_coefficients!(c, z, tolerance=eps(T)) -> c

In-place conversion of samples in `c` to interpolation coefficients for a B-spline
whose poles are given by `z`.

Optional `tolerance` is the admissible relative error; `T` is the floating-point of the
elements of `c` and of the poles in `z`.

"""
function to_interpolation_coefficients!(
    c::AbstractVector{T},   # input samples --> output coefficients
    z::BSplinePoles{D,T,N}, # poles
    tolerance::T = eps(T),  # admissible relative error
    ) where {D,T<:AbstractFloat,N}

    start, stop = index_bounds(c)
    if start ≥ stop
        # mirror boundaries require at least 2 samples
        return c
    end

    # compute the overall gain (FIXME can be pre-computed once)
    λ = T(𝟙)
    for zₖ in values(z)
        λ *= (𝟙 - zₖ)*(𝟙 - 𝟙/zₖ)
    end

    # apply the gain (FIXME the gain may be applied in the 1st (causal) pass)
    @inbounds @simd for i in start : stop
        c[i] *= λ
    end

    # loop over all poles
    @inbounds for zₖ in values(z)
        # causal initialization
        c[start] = initial_causal_coefficient(c, zₖ, tolerance)
        # causal recursion
        @simd for i in start + 1 : stop
            c[i] += zₖ*c[i - 1]
        end
        # anticausal initialization
        c[start] = initial_anticausal_coefficient(c, zₖ)
        # anticausal recursion
        @simd for i in stop - 1 : -1 : start
            c[i] = zₖ*(c[i + 1] - c[i])
        end
    end
    return c
end

# this initialization corresponds to mirror boundaries
# FIXME pass log(tolerance) for speed
function initial_causal_coefficient(
    c::AbstractVector{T}, # coefficients
    z::T,                 # actual pole
    tolerance::T,         # admissible relative error
    ) where {T<:AbstractFloat}

    start, stop = index_bounds(c)
    horizon = length(c)::Int
    if tolerance > 𝟘
        horizon = ceil(Int, log(tolerance)/log(abs(z))) # FIXME use log2 if faster
    end
    if horizon < length(c)
        # accelerated loop
        zn = z
        sum = c[start]
        for i in start+1 : start+horizon-1 # FIXME check bounds
            sum += zn*c[i]
            zn *= z
        end
        return sum
    else
        # full loop
        zn = z
        iz = 𝟙/z
        z2n = z^(length(c) - 1)
        sum = c[start] + z2n*c[stop]
        z2n *= z2n*iz
        for i in start+1 : stop-1 # FIXME check bounds
            sum += (zn + z2n)*c[i]
            zn *= z
            z2n *= iz
        end
        return sum/(𝟙 - zn*zn)
    end
end

# this initialization corresponds to mirror boundaries
function initial_anticausal_coefficient(
    c::AbstractVector{T}, # coefficients
    z::T,                 # actual pole
    ) where {T<:AbstractFloat}

    return (z/(z*z - 𝟙))*(z*c[end - 1] + c[end])
end
