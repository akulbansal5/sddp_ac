"""
code for generating data for GEP
"""

using Distributions, Serialization, CSV, DataFrames

# const HORIZON = 5
const GENERATOR_LIST = Dict(1 => "BaseLoad", 2 => "CC", 3 => "CT", 4 => "Nuclear", 5 => "Wind", 6 => "IGCC")
const MAX_OUTPUT = [1130.0, 390.0, 380.0, 1180.0, 175.0, 560.0]
const MAX_UNIT = [4, 10, 10, 1, 45, 4]
const SUBPERIOD = 3
const SUBPERIOD_HOUR = [271.0, 6556.0, 1933.0]
const CONSTRUCTION_COST = [1.446, 0.795, 0.575, 1.613, 1.650, 1.671]
const MAX_CAPACITY = [1200.0, 400.0, 400.0, 1200.0, 500.0, 600.0]
const FUEL_PRICE = [3.37, 9.37, 9.37, 0.93e-3, 0.0, 3.37] .* 1e-6
const RATIO = [8.844 / 0.4, 7.196 / 0.56, 10.842 / 0.4, 10.400 / 0.45, 0.0, 8.613 / 0.48]
const FUEL_PRICE_GROWTH = 0.02
const OPERATING_COST = [4.7, 2.11, 3.66, 0.51, 5.00, 2.98] .* 1e-6
const OPER_COST_GROWTH = 0.03
const PENALTY_COST = 1e-1
const LAMBDA = [1.38, 1.04, 0.80]
const HOURS_PER_YEAR = 8760.0
const rate = 0.08

#price for natural gas is with unit of 1000 cubic feet
const price_mean = [9.37000, 9.79257, 10.23477, 10.69770, 11.18226, 11.68951, 12.22057, 12.77657, 13.35693, 13.96636, 14.21, 14.52]
const price_std = [0.0, 0.9477973, 1.2146955, 1.4957925, 1.7848757, 2.0798978, 2.3814051, 2.6911019, 3.0063683, 3.3382216, 3.56, 3.81]
const D0 = 0.57

#code for scenario tree generation
#Dict: stage -> Vector of uncertainty realizations


# print(typeof(Dict(i => [[] for j in 1:3] for i in 1:5)))

function scenario_gen(scens, HORIZON)
    scen_count = Dict(i => scens for i in 2:HORIZON)
    scen_count[1] = 1
    scenarios = Dict(i => [[] for j in 1:scen_count[i]] for i in 1:HORIZON)

    #the demand is random and that is what we look at.
    #for each time period
    truncated_normal_dist = Truncated(Normal(0, 1), -1, 1)
    for t in 1:HORIZON
        dem_coeff = D0*(1.015^t - 1.005^t)*rand(scens) .+ D0*1.005^t
        
        # dem_coeff   = D0*(1.010^t)*rand(scens) .+ D0*1.005^t
        #generate the number of scenarios as in scen_count
        # println("demand coefficients at stage $(t): $(dem_coeff)")
        for j in 1:scen_count[t]
            # println(maximum([dem_coeff[j] - D0 ,0.0])*1e9)
            sub_dems        = LAMBDA*maximum([dem_coeff[j] - D0 ,0.0])*1e9/HOURS_PER_YEAR
            gas_price       = price_mean[t] .+ price_std[t] * rand(truncated_normal_dist, 1) * 1e-6 / 1.028
            scen            = vcat(sub_dems, gas_price)
            scenarios[t][j] = scen
        end
    end
    return scenarios
end

#serialize the scenarios data
function instGen(instList, folder, prefix)

    for inst in instList
        id = inst[1]
        rep = inst[2]
        stages = inst[3]
        scens = inst[4]
        for j in 1:Int64(rep)
            support = scenario_gen(scens, stages)
            file = folder*prefix*"_$(id)_$(j)_$(stages)_$(scens).jls"
            println("_$(id)_$(j)_$(stages)_$(scens).jls")
            # println("support: ", support)
            open(file, "w") do f
                serialize(f, support)
            end
        end
    end
end


function instGen_csv(csvfile, instIDList, folder, prefix)
    """
    generates instances using IDs in the csv file
    """
    allInst = []
    df = CSV.File(csvfile) |> DataFrame
    filter_df = df[in(instIDList).(df.id), :]
    for row in eachrow(filter_df)
        push!(allInst, [row.id, row.rep, row.T, row.scens])
    end
    println(allInst)
    instGen(allInst, folder, prefix)
end



# num_stages = 3
# num_scenarios = 3
folder = joinpath(@__DIR__, "gep_data") * "/"
prefix = "gep"
csv_file = folder*"gep_instances.csv"
instGen_csv(csv_file, 58:59, folder, prefix)



