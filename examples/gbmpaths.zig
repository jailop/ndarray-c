const std = @import("std");
const ndarray = @import("ndarray");

/// Geometric Brownian Motion parameters
const GBMParams = struct {
    mu: f64, // drift coefficient
    sigma: f64, // volatility coefficient
};

/// Simulation parameters
const SimParams = struct {
    T: f64, // time to maturity
    M: i32, // number of steps by period
    I: i32, // simulation paths
};

/// Generates Geometric Brownian Motion paths using the Zig ndarray wrapper
///
/// Parameters:
/// - gbmPar: GBM parameters (mu, sigma)
/// - simPar: Simulation parameters (T, M, I)
/// - S0: Initial stock price
///
/// Returns: Array of GBM paths with shape [steps, I]
///
/// The algorithm implements the stochastic differential equation:
/// dS = mu*S*dt + sigma*S*dW
/// where dW is a Wiener process increment
pub fn generateGBMPaths(
    allocator: std.mem.Allocator,
    gbmPar: GBMParams,
    simPar: SimParams,
    S0: f64,
) !ndarray.NDArray {
    // Auxiliary variables
    const steps = simPar.M * @as(i32, @intFromFloat(simPar.T)) + 1;
    const dt = 1.0 / @as(f64, @floatFromInt(simPar.M));
    const sdt = @sqrt(dt);
    const drift = (gbmPar.mu - 0.5 * gbmPar.sigma * gbmPar.sigma) * dt;

    // Prices array: [steps, I]
    var S = try ndarray.NDArray.initZeros(&[_]usize{ @intCast(steps), @intCast(simPar.I) });

    // Set initial price for all paths
    const initial_prices = try allocator.alloc(f64, @intCast(simPar.I));
    defer allocator.free(initial_prices);
    @memset(initial_prices, S0);
    _ = S.setSlice(0, 0, initial_prices);

    // Wiener process: [steps, I]
    var z = try ndarray.NDArray.initRandomNormal(&[_]usize{ @intCast(steps), @intCast(simPar.I) }, 0.0, 1.0);
    defer z.deinit();

    // Temporary arrays for single time step: [1, I]
    var S_t = try ndarray.NDArray.init(&[_]usize{ 1, @intCast(simPar.I) });
    defer S_t.deinit();
    var z_t = try ndarray.NDArray.init(&[_]usize{ 1, @intCast(simPar.I) });
    defer z_t.deinit();

    // Step through time evolution
    var step: i32 = 1;
    while (step < steps) : (step += 1) {
        // z_t = z[step, :] - copy current time step
        _ = z_t.copySlice(0, 0, z, 0, @intCast(step));

        // z_t = z_t * sigma * sdt + drift
        _ = z_t.mulScalar(gbmPar.sigma * sdt).addScalar(drift);

        // z_t = exp(z_t)
        _ = z_t.exp();

        // S_t = S[step-1, :] - copy previous time step prices
        _ = S_t.copySlice(0, 0, S, 0, @intCast(step - 1));

        // S_t = S_t * z_t - apply geometric growth factor
        _ = S_t.mul(z_t);

        // S[step, :] = S_t - store current time step prices
        _ = S.copySlice(0, @intCast(step), S_t, 0, 0);
    }

    return S;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // GBM parameters
    const gbmParams = GBMParams{
        .mu = 0.05, // 5% annual drift
        .sigma = 0.2, // 20% annual volatility
    };

    // Simulation parameters
    const simParams = SimParams{
        .T = 1.0, // 1 year to maturity
        .M = 252, // trading days per year
        .I = 10000, 
    };

    const S0 = 100.0; // Initial stock price

    // Generate GBM paths
    const paths = try generateGBMPaths(allocator, gbmParams, simParams, S0);
    defer paths.deinit();

    // Print the generated paths
    paths.print("Prices", 4);

    // Calculate some basic statistics - take the last time step
    const final_step = simParams.M * @as(i32, @intFromFloat(simParams.T));
    const final_prices = try paths.initTake(0, @intCast(final_step), @intCast(final_step + 1));
    defer final_prices.deinit();

    const mean_final_price = final_prices.scalarAggregate(.mean);
    const std_final_price = final_prices.scalarAggregate(.std);
    const min_final_price = final_prices.scalarAggregate(.min);
    const max_final_price = final_prices.scalarAggregate(.max);

    std.debug.print("Final price statistics:\n", .{});
    std.debug.print("  Mean:     {d:.4}\n", .{mean_final_price});
    std.debug.print("  Std Dev:  {d:.4}\n", .{std_final_price});
    std.debug.print("  Min:      {d:.4}\n", .{min_final_price});
    std.debug.print("  Max:      {d:.4}\n", .{max_final_price});

    // Calculate theoretical expected value and variance for GBM
    const expected_S_T = S0 * std.math.exp(gbmParams.mu * simParams.T);
    const var_S_T = S0 * S0 * std.math.exp(2 * gbmParams.mu * simParams.T) *
        (std.math.exp(gbmParams.sigma * gbmParams.sigma * simParams.T) - 1);
    const std_S_T = std.math.sqrt(var_S_T);

    std.debug.print("\nTheoretical values for GBM(S_T):\n", .{});
    std.debug.print("  Expected value: {d:.4}\n", .{expected_S_T});
    std.debug.print("  Standard deviation: {d:.4}\n", .{std_S_T});
}
