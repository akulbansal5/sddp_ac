"""
data generation for stochastic multiple binary knapsack problem (SMKP)
ref: Angulo et al (2016): Improving the Integer L-Shaped Method


A: number of instances, number of stages, number of rows, columns
C: number of instances, number of stages, number of rows, columns
T: number of instances, number of stages, number of rows, columns
c: number of instances, number of stages, number of columns
d: number of instances, number of stages, number of columns
q: number of instances, number of stages, number of scenarios per node, number of columns

The files are saved in the smkp_data/ folder relative to this script.
"""

using CSV, DataFrames





using Serialization, Random

values = range(1, 100)
# Create smkp_data folder if it doesn't exist
folder = joinpath(@__DIR__, "smkp_data")
mkdir(folder; exist_ok = true)


function Agen(id, inst, st, rows, cols, scens, prefix)

    """
    generates A matrix associated with state variables
    inst (int): number of instances -> generates instances for same class
    st (int): number of stages
    rows: number of rows of the matrix
    cols: number of columns of the matrix
    prefix: prefix used while printing filename
    """

    for i in inst
        for t in 1:st
            A = rand(values, rows, cols)
            # println(A)
            file = prefix*"_$(id)_$(i)_$(t)_$(rows)_$(cols)_$(scens).jls"
            open(joinpath(folder, file), "w") do f
                serialize(f, A)
            end
        end
    end
end

function qGen(id, inst, st, scens, cols, prefix)
    """
    generates the cost coefficients associated with smkp problem_class

    inst:  sets of instances -> generates instances for same class
    st:    number of stages
    scens: number of scenarios
    cols:  number of columns in the q vector
    """

    for i in inst
        for t in 1:st
            for s in 1:scens
                
                q = rand(values, cols)
                file = prefix*"_$(id)_$(i)_$(t)_$(s)_$(cols).jls"
                open(joinpath(folder, file), "w") do f
                    serialize(f, q)
                end
            end
        end
    end
end


# Generate A matrix one for each stage
# allInst = [[1, 10, 5, 120, 5], [2, 4, 5, 120, 20], [3, 5, 10, 120, 10], [4, 10, 10, 120, 2], [5, 5, 20, 120, 5]]
# allInst = [[6, 5, 10, 240, 10], [7, 5, 10, 60, 10]]
# allInst = [[8, 3, 10, 120, 5], [9, 3, 10, 120, 10], [10, 3, 10, 120, 15], [11, 4, 5, 120, 5], [12, 4, 10, 120, 5]]
# allInst = [[13,4,25,120,5], [16,4,50,120,10], [17,4,25,60,5], [20,4,50,60,10], [21,8,25,120,5]]
# allInst = [[29, 3, 20, 60, 5], [30, 3, 100, 60, 2], [31, 4, 20, 60, 3], [32, 2, 25, 60, 10], [33, 3, 50, 120, 2]]
# allInst = [[15,4,50,120,5],[22,8,25,120,10],[23,8,50,120,5],[24,8,50,120,10]]
# allInst = [[34, 4, 500, 1200, 5], [35, 4, 800, 1600, 5], [36, 4, 500, 1200, 10], [37, 4, 800, 1600, 10]]
# allInst   = [[37, 4, 800, 1600, 10]]


function instGen(allInst)


    for pickInst in allInst

        

        inst  = [1]
        id    = pickInst[1]
        st    = pickInst[2]
        rows  = pickInst[3]
        cols  = pickInst[4]
        scens = pickInst[5]

        #Generate A matrix for each stage
        Agen(id, inst, st, rows, cols, scens, "smkp_A")

        # Generate T matrix one for each stage
        Agen(id, inst, st, rows, cols, scens, "smkp_T")

        # Generate c vector
        qGen(id, inst, 1, 1, cols, "smkp_c")

        #generate d vector
        qGen(id, inst, 1, 1, cols, "smkp_d")

        # Generate two scenarios per stage
        qGen(id, inst,st, scens, cols, "smkp_q")

        println("Instance generated for ", pickInst)

    end
end


filename = joinpath(@__DIR__, "smkp_data", "instances.csv")
function instGen_csv(csvfile, instIDList)
    """
    generates instances using IDs in the csv file
    """
    allInst = []
    df = CSV.File(csvfile) |> DataFrame
    filter_df = df[in(instIDList).(df.id), :]
    for row in eachrow(filter_df)
        push!(allInst, [row.id, row.T, row.rows, row.cols, row.scens])
    end
    print(allInst)
    instGen(allInst)
end

# instStart = 96
# instEnd = 116
# instIDList = [9,10,12,44,45, 46, 49, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63]
# instGen_csv(filename, instStart:instEnd)

instIDList = 249:518
instGen_csv(filename, instIDList)
