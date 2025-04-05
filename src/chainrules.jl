using GeometricUnits
using ChainRulesCore
using Zygote

dimensionless_zero(a::GQ) = dimensionless(zero(value(a)))
dimensionless_unit(a::GQ) = dimensionless(oneunit(value(a)))

function ChainRulesCore.rrule(::Type{GQ{d,X}}, x::X) where {d,X<:AbstractFloat}
    y = GQ{d}(x)
    function _pullback(ybar)
        fbar = NoTangent()
        xbar = oneunit(y) * ybar
        return fbar, xbar
    end
    return y, _pullback
end

(p::ProjectTo{X})(a::GQ) where {X<:AbstractFloat} = X(a.x)

Base.@eval Zygote._gradcopy!(dst::AbstractArray, src::GQ) = copyto!(dst, src)

function ChainRulesCore.rrule(::typeof(oneunit), a::GQ)
    y = oneunit(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = dimensionless_zero(a) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(zero), a::GQ)
    y = zero(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = dimensionless_zero(a) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(oftype), a::GQ, x)
    y = oftype(a, x)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = dimensionless_zero(a) * ybar
        xbar = oftype(a, oneunit(value(y))) * ybar
        return fbar, abar, xbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(value), a::GQ)
    y = value(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = 1 / oneunit(a) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(dimensionless), x)
    y = dimensionless(x)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = oneunit(y) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(time), x)
    y = time(x)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = oneunit(y) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(frequency), x)
    y = frequency(x)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = oneunit(x) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(+), a::GQ, b::GQ)
    y = a + b
    function _pullback(ybar)
        fbar = NoTangent()
        abar = @thunk(dimensionless_unit(a) * ybar)
        bbar = @thunk(dimensionless_unit(b) * ybar)
        return fbar, abar, bbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(-), a::GQ, b::GQ)
    y = a - b
    function _pullback(ybar)
        fbar = NoTangent()
        abar = @thunk(dimensionless_unit(a) * ybar)
        bbar = @thunk(-dimensionless_unit(b) * ybar)
        return fbar, abar, bbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(-), a::GQ)
    y = -a
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -dimensionless_unit(a) * ybar
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(*), a::GQ, b::GQ)
    y = a * b
    function _pullback(ybar)
        fbar = NoTangent()
        abar = @thunk(b * ybar)
        bbar = @thunk(a * ybar)
        return fbar, abar, bbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(/), a::GQ, b::GQ)
    y = a / b
    function _pullback(ybar)
        fbar = NoTangent()
        abar = @thunk(ybar / b)
        bbar = @thunk(-ybar * a / b / b)
        return fbar, abar, bbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(^), a::GQ, b::GQ)
    y = a^b
    function _pullback(ybar)
        fbar = NoTangent()
        abar = @thunk(ybar * b * y / a)
        bbar = @thunk(ybar * y * log(a))
        return fbar, abar, bbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(^), a::GQ, n::Integer)
    y = a^n
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * signed(n) * y / a
        nbar = NoTangent()
        return fbar, abar, nbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(^), a::GQ, n::Rational)
    y = a^n
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * n * y / a
        nbar = NoTangent()
        return fbar, abar, nbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(sqrt), a::GQ)
    y = sqrt(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * (1 // 2) * y / a
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(cbrt), a::GQ)
    y = cbrt(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * (1 // 3) * y / a
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(root), a::GQ, n::Integer)
    y = root(a, n)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * (1 // n) * y / a
        nbar = NoTangent()
        return fbar, abar, nbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(root), a::GQ, n::Rational)
    y = root(a, n)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * (1 // n) * y / a
        nbar = NoTangent()
        return fbar, abar, nbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(exp), a::GQ)
    y = exp(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * y
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(exp10), a::GQ)
    y = exp10(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * y * log(10)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(exp2), a::GQ)
    y = exp2(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * y * log(2)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(log), a::GQ)
    y = log(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar / a
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(log2), a::GQ)
    y = log2(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar / a / log(2)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(log10), a::GQ)
    y = log10(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar / a / log(10)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(sin), a::GQ)
    y = sin(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * cos(a)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(cos), a::GQ)
    y = cos(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -ybar * sin(a)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(sincos), a::GQ)
    sina, cosa = sincos(a)
    function _pullback(ybar)
        ybar1, ybar2 = ybar
        fbar = NoTangent()
        abar = ybar1 * cosa - ybar2 * sina
        return fbar, abar
    end
    return (sina, cosa), _pullback
end

function ChainRulesCore.rrule(::typeof(tan), a::GQ)
    y = tan(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * sec(a)^2
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(csc), a::GQ)
    y = csc(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -ybar * cot(a) * y
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(sec), a::GQ)
    y = sec(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar * tan(a) * y
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(cot), a::GQ)
    y = cot(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -ybar * csc(a)^2
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(asin), a::GQ)
    y = asin(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar / sqrt(1 - a * a)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(acos), a::GQ)
    y = acos(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -ybar / sqrt(1 - a * a)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(atan), a::GQ)
    y = atan(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar / (1 + a * a)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(atan), a::GQ, b::GQ)
    y = atan(a, b)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = @thunk(ybar * b / (a * a + b * b))
        bbar = @thunk(-ybar * a / (a * a + b * b))
        return fbar, abar, bbar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(acsc), a::GQ)
    y = acsc(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -ybar / a / sqrt(a * a - 1)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(asec), a::GQ)
    y = asec(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = ybar / a / sqrt(a * a - 1)
        return fbar, abar
    end
    return y, _pullback
end

function ChainRulesCore.rrule(::typeof(acot), a::GQ)
    y = acot(a)
    function _pullback(ybar)
        fbar = NoTangent()
        abar = -ybar / (1 + a * a)
        return fbar, abar
    end
    return y, _pullback
end

# function ChainRulesCore.rrule(::typeof(taylor_horner), x::GQ, cs)
#     y = taylor_horner(x, cs)
#     function _pullback(ybar)
#         fbar = NoTangent()

#         result = GQ(zero(cs[n].x), cs[n].d - x.d)
#         @inbounds for ii = n:-1:1
#             result = result * x / ii + cs[ii]
#         end
#     end
# end
