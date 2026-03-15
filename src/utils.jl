# ══════════════════════════════════════════════════════════════════
#  utils.jl — Metrics, plotting utilities
# ══════════════════════════════════════════════════════════════════

"""
Regression metrics: MSE, MAE, R² (scalar output).
"""
function computeMetrics(ŷ::AbstractMatrix, y::AbstractMatrix)
    residuals = ŷ .- y

    mseVal = mean(residuals .^ 2)
    maeVal = mean(abs.(residuals))

    ssTot = sum((y .- mean(y)) .^ 2)
    ssRes = sum(residuals .^ 2)
    r2Val = 1.0 - ssRes / ssTot

    return (mse=mseVal, mae=maeVal, r2=r2Val)
end

"""
Convert (r, θ) to Cartesian sector-plot coordinates:
  x = r * sin(θ − θ_center)
  y = r * cos(θ − θ_center)
"""
function makeSectorData(r, theta, thetaCenter)
    x = r .* sin.(theta .- thetaCenter)
    y = r .* cos.(theta .- thetaCenter)
    return x, y
end
