# ══════════════════════════════════════════════════════════════════
#  graph.jl — Delaunay tessellation → neighbor graph
#
#  Builds a fixed-degree graph where each node has exactly 4
#  neighbors (the 4 Delaunay neighbors), sorted by ascending distance.
# ══════════════════════════════════════════════════════════════════

"""
Convert Cartesian positions (3 × N) to spherical coordinates
relative to an observer.

Returns three vectors of length N:
- r   : radial distance from observer
- theta : polar angle from +z axis  (0 = along +z)
- phi   : azimuthal angle in the x-y plane
"""
function toSpherical(positions::Matrix{Float64}, observer::Vector{Float64}=[0.0, 0.0, 0.0])
    n = size(positions, 2)
    r = Vector{Float64}(undef, n)
    theta = Vector{Float64}(undef, n)
    phi = Vector{Float64}(undef, n)

    for i in 1:n
        dx = positions[1, i] - observer[1]
        dy = positions[2, i] - observer[2]
        dz = positions[3, i] - observer[3]

        r[i] = sqrt(dx^2 + dy^2 + dz^2)
        theta[i] = acos(dz / r[i])
        phi[i] = atan(dy, dx)
    end
    return r, theta, phi
end

"""
Run TetGen Delaunay tetrahedralization on a 3 × N point matrix.

Returns:
- tets   : M × 4 tetrahedron index matrix (1-based)
"""
function buildTessellation(points::Matrix{Float64})
    meshdata = TetGen.RawTetGenIO{Float64}()
    meshdata.pointlist = points
    result = TetGen.tetrahedralize(meshdata, "Q")
    return Matrix(result.tetrahedronlist')
end

"""
Extract the full adjacency from a tetrahedralization.
Returns a vector of neighbor lists (one per point).
"""
function extractNeighbors(tets::AbstractMatrix{<:Integer}, nPoints::Int)
    adj = [Set{Int}() for _ in 1:nPoints]

    for row in eachrow(tets)
        #v = ((row[1]), Int(row[2]), Int(row[3]), Int(row[4]))
        for a in 1:4, b in 1:4
            a == b && continue
            push!(adj[row[a]], row[b])
        end
    end

    return [sort(collect(s)) for s in adj]
end


"""
From a full adjacency list, select the 4 nearest neighbors for
each node (sorted by ascending distance).

Returns a `4 × N` matrix where `neighborIdx[j, i]` is the j-th
nearest Delaunay neighbor of node `i`.

"""
function buildNeighborIdx(points::Matrix{Float64}, neighbors::Vector{Vector{Int}})

    k = 4 # number of neighbors
    n = size(points, 2)
    neighborIdx = zeros(Int, k, n)

    for i in 1:n
        nbrs = neighbors[i] #the neighbors of node i

        # Compute distances to each neighbor and sort them
        dists = [norm(@view(points[:, j]) .- @view(points[:, i])) for j in nbrs]
        order = sortperm(dists)

        for j in 1:k
            neighborIdx[j, i] = nbrs[order[j]]
        end
    end

    return neighborIdx
end

"""
End-to-end pipeline: tessellate → extract neighbors → sort by distance
→ convert to spherical coordinates → assemble feature matrix.

Returns:
- features    : 3 × N Float32 matrix  (r_obs, θ, φ)
- neighborIdx : 4 × N Int matrix       (sorted neighbor indices)
"""
function buildGraph(points::Matrix{Float64}, observer::Vector{Float64})
    # Tessellate
    tets = buildTessellation(points)
    nPoints = size(points, 2)

    # Full adjacency
    neighbors = extractNeighbors(tets, nPoints)

    # 4 nearest neighbors per node, sorted by distance
    neighborIdx = buildNeighborIdx(points, neighbors)

    # Spherical coordinates relative to observer
    r, theta, phi = toSpherical(points, observer)

    # Feature matrix: (r_obs, θ, φ)
    features = Float32.(vcat(r', theta', phi'))   # 3 × N

    return features, neighborIdx
end
