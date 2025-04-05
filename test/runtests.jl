using GeometricUnitsChainRules
using BenchmarkTools
using Test

@testset "GeometricUnitsChainRules.jl" verbose = true begin
    @testset "derivatives" begin
        d1 = dimensionless(1.2)
        d2 = dimensionless(2.1)
        d3 = dimensionless(0.5)
        t1 = time(1.3)
        t2 = time(2.4)
        x = 0.9

        @test gradient(a -> value(oneunit(a)), d1) == (dimensionless(0.0),)
        @test @ballocated(
            gradient($(a -> value(oneunit(a))), $d1) == ($dimensionless(0.0),)
        ) == 0

        @test gradient(a -> value(zero(a)), d1) == (dimensionless(0.0),)
        @test @ballocated(
            gradient($(a -> value(zero(a))), $d1) == ($dimensionless(0.0),)
        ) == 0

        @test gradient((t, x) -> value(oftype(t, x)), t1, x) == (zero(1 / t1), 1)
        @test @ballocated(gradient((t, x) -> value(oftype(t, x)), $t1, $x)) == 0

        @test gradient(d -> value(d * d - 2 * d + 1), d1) == (2 * d1 - 2,)
        @test @ballocated(gradient(d -> value(d * d - 2 * d + 1), $d1)) == 0

        @test gradient(d -> value(-d + exp(d)), d1) == (-oneunit(d1) + exp(d1),)
        @test @ballocated(gradient(d -> value(-d + exp(d)), $d1)) == 0

        @test gradient(d -> value((d + 1) / d), d1) == gradient(d -> value(1 + 1 / d), d1)
        @test gradient(d -> value((d + 1) / d), d1)[1] ≈ -1 / d1^2
        @test @ballocated(gradient(d -> value((d + 1) / d), $d1)) == 0
        @test @ballocated(gradient(d -> value(1 + 1 / d), $d1)) == 0

        @test gradient((a, b) -> value(a^b), d1, d2) ==
              gradient((a, b) -> value(exp(b * log(a))), d1, d2)
        @test gradient(a -> value(exp10(a)), d1)[1] ≈
              gradient(a -> value(exp(a * log(10))), d1)[1]
        @test gradient(a -> value(exp2(a)), d1)[1] ≈
              gradient(a -> value(exp(a * log(2))), d1)[1]
        @test @ballocated(gradient((a, b) -> value(a^b), $d1, $d2)) == 0
        @test @ballocated(gradient((a, b) -> value(exp(b * log(a))), $d1, $d2)) == 0

        @test gradient(a -> value(a^2.0), d1) == gradient(a -> value(a * a), d1)
        @test @ballocated(gradient(a -> value(a^2.0), $d1)) == 0

        @test gradient(a -> value(a^2), d1) == gradient(a -> value(a * a), d1)
        @test_broken @ballocated(gradient(a -> value(a^2), $d1)) == 0

        @test gradient(a -> value(a^unsigned(2)), d1) == gradient(a -> value(a * a), d1)
        @test_broken @ballocated(gradient(a -> value(a^unsigned(2)), $d1)) == 0

        @test gradient(a -> value(a^(3 // 1)), d1) == gradient(a -> value(a * a * a), d1)
        @test @ballocated(gradient(a -> value(a^(3 // 1)), $d1)) == 0

        @test gradient(a -> value(sqrt(a * a * a)), d1) ==
              gradient(a -> value(a^(3 // 2)), d1)
        @test @ballocated(gradient(a -> value(sqrt(a * a * a)), $d1)) == 0

        @test gradient(a -> value(cbrt(sqrt(a))), d1) ==
              gradient(a -> value(root(a, 6)), d1)
        @test @ballocated(gradient(a -> value(cbrt(sqrt(a))), $d1)) == 0
        @test @ballocated(gradient(a -> value(root(a, 6)), $d1)) == 0

        @test gradient(a -> value(root(a, 2 // 3)), d1) ==
              gradient(a -> value(sqrt(a * a * a)), d1)
        @test @ballocated(gradient(a -> value(root(a, 2 // 3)), $d1)) == 0

        @test gradient(a -> value(log2(a)), d1) == gradient(a -> value(log(a) / log(2)), d1)
        @test @ballocated(gradient(a -> value(log2(a)), $d1)) == 0

        @test gradient(a -> value(log10(2 * a)), d1) ==
              gradient(a -> value(log(a) / log(10)), d1)
        @test @ballocated(gradient(a -> value(log10(2 * a)), $d1)) == 0

        @test gradient(a -> value(tan(a)), d1)[1] ≈
              gradient(a -> value(sin(a) / cos(a)), d1)[1]
        @test @ballocated(gradient(a -> value(tan(a)), $d1)) == 0
        @test @ballocated(gradient(a -> value(sin(a) / cos(a)), $d1)) == 0

        @test gradient(a -> value(sec(a)), d1)[1] ≈ gradient(a -> value(1 / cos(a)), d1)[1]
        @test @ballocated(gradient(a -> value(sec(a)), $d1)) == 0

        @test gradient(a -> value(csc(a)), d1)[1] ≈ gradient(a -> value(1 / sin(a)), d1)[1]
        @test @ballocated(gradient(a -> value(csc(a)), $d1)) == 0

        @test gradient(a -> value(cot(a)), d1)[1] ≈
              gradient(a -> value(cos(a) / sin(a)), d1)[1]
        @test @ballocated(gradient(a -> value(cot(a)), $d1)) == 0

        @test gradient(a -> value(asin(a)), d3)[1] == -gradient(a -> value(acos(a)), d3)[1]
        @test @ballocated(gradient(a -> value(asin(a)), $d3)) == 0
        @test @ballocated(gradient(a -> value(acos(a)), $d3)) == 0

        @test gradient(a -> value(atan(sin(a) / cos(a))), d1)[1] == oneunit(d1)
        @test gradient(a -> value(atan(sin(a), cos(a))), d1)[1] == oneunit(d1)
        @test @ballocated(gradient(a -> value(atan(sin(a) / cos(a))), $d1)) == 0
        @test_broken @ballocated(gradient(a -> value(atan(sin(a), cos(a))), $d1)) == 0

        @test gradient(a -> value(asin(sin(a))), d3)[1].x ≈ 1
        @test gradient(a -> value(acos(cos(a))), d3)[1].x ≈ 1
        @test gradient(a -> value(atan(tan(a))), d3)[1].x ≈ 1
        @test gradient(a -> value(acsc(csc(a))), d3)[1].x ≈ 1
        @test gradient(a -> value(asec(sec(a))), d3)[1].x ≈ 1
        @test gradient(a -> value(acot(cot(a))), d3)[1].x ≈ 1

        a = dimensionless(1.1)
        (s, c), _back = pullback(sincos, a)
        @test _back((1, 0))[1] == c
        @test _back((0, 1))[1] == -s
        @test @ballocated(gradient(a -> value(sum(sincos(a))), $a)) == 0

        function func1(a, b, c, t)
            qt = time(t)
            qc = dimensionless(c)
            qb = frequency(b)
            qa = GQ{-2}(a)
            return value(qa * qt * qt + qb * qt + qc)
        end

        function func1_grad_anl(a, b, c, t)
            return t * t, t, 1.0, 2 * a * t + b
        end

        @test collect(func1_grad_anl(1.0, 2.0, 3.0, -2.0)) ≈
              collect(gradient(func1, 1.0, 2.0, 3.0, -2.0))

        function func2(a, w, t)
            qa = dimensionless(a)
            qw = frequency(w)
            qt = time(t)
            return [value(qa * sin(qw * qt)), value(qa * cos(qw * qt))]
        end

        function func2_jac_anl(a, w, t)
            return (
                [sin(w * t), cos(w * t)],
                [a * t * cos(w * t), -a * t * sin(w * t)],
                [a * w * cos(w * t), -a * w * sin(w * t)],
            )
        end

        jac1 = jacobian(func2, Float128(1.2), Float128(0.5), Float128(2.3))
        jac2 = func2_jac_anl(Float128(1.2), Float128(0.5), Float128(2.3))
        @test all([j1 ≈ j2 for (j1, j2) in zip(jac1, jac2)])
    end
end
