using Test
using JuMP
import MathOptInterface as MOI
import Gurobi
using ProjectedConeOracle

@testset "Solver adapter" begin
    @testset "HiGHSAdapter" begin
        adapter = HiGHSAdapter()
        model = create_model(adapter)
        backend = JuMP.backend(model)

        @test MOI.get(backend, MOI.Silent()) == true
        @test MOI.get(backend, MOI.RawOptimizerAttribute("solver")) == "ipm"
        @test_throws MOI.UnsupportedAttribute MOI.get(backend, MOI.RawOptimizerAttribute("Method"))

        configure_verified!(adapter, model)
        @test MOI.get(backend, MOI.RawOptimizerAttribute("solver")) == "simplex"

        configure_recovery!(adapter, model)
        @test MOI.get(backend, MOI.RawOptimizerAttribute("solver")) == "simplex"

        configure_default!(adapter, model)
        @test MOI.get(backend, MOI.RawOptimizerAttribute("solver")) == adapter.solver
        @test supports_two_phase(adapter)
    end

    @testset "GurobiAdapter (if available)" begin
        withenv("GRB_LICENSE_FILE" => get(ENV, "GRB_LICENSE_FILE", "/opt/licenses/gurobi.lic")) do
            try
                adapter = GurobiAdapter()
                model = create_model(adapter)
                backend = JuMP.backend(model)
                @test MOI.get(backend, MOI.Silent()) == true
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Method")) == 1
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Presolve")) == 0
                configure_fast!(adapter, model)
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Method")) == 0
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Presolve")) == 0
                configure_verified!(adapter, model)
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Method")) == 1
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Presolve")) == 0
                configure_default!(adapter, model)
                @test MOI.get(backend, MOI.RawOptimizerAttribute("Method")) == 1
            catch err
                @info "Skipping active Gurobi assertions" exception=(err, catch_backtrace())
            end
        end
    end
end
