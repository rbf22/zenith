from std.math import exp, sin
from std.memory import stack_allocation
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime P = 64
comptime N = 128
comptime B = 4
comptime T = 512
comptime H = 20
comptime NX = B * T * H * P
comptime NS = B * T * H * N
comptime NH = B * T * H


def gen(n: Int, salt: Int) -> Float64:
    return sin(Float64(n * 131 + salt * 7919) * 0.001)


def make_inputs(
    mut x: List[Float32], mut q: List[Float32], mut k: List[Float32], mut kp: List[Float32],
    mut xp: List[Float32], mut adt: List[Float32], mut gamma: List[Float32], mut beta: List[Float32],
):
    for i in range(NX):
        x[i] = Float32(gen(i, 1))
        xp[i] = Float32(gen(i, 2))
    for i in range(NS):
        q[i] = Float32(gen(i, 3))
        k[i] = Float32(gen(i, 4))
        kp[i] = Float32(gen(i, 5))
    for i in range(NH):
        adt[i] = Float32(-0.05 * (1.0 + gen(i, 6)))
        gamma[i] = Float32(0.05 * (1.0 + gen(i, 7)))
        beta[i] = Float32(0.05 * (1.0 + gen(i, 8)))


def ssd_fwd_cpu(
    x: List[Float32], q: List[Float32], k: List[Float32], kp: List[Float32], xp: List[Float32],
    adt: List[Float32], gamma: List[Float32], beta: List[Float32], mut y: List[Float32],
):
    for bh in range(B * H):
        var b = bh // H
        var h = bh % H
        for p in range(P):
            var S = List[Float32](length=N, fill=0.0)
            for t in range(T):
                var base = (b * T + t) * H + h
                var a = exp(adt[base])
                var yv: Float32 = 0.0
                for n in range(N):
                    S[n] = a * S[n] + gamma[base] * x[base * P + p] * k[base * N + n] + beta[base] * xp[base * P + p] * kp[base * N + n]
                    yv += S[n] * q[base * N + n]
                y[base * P + p] = yv


def ssd_fwd_gpu(
    x: Pointer[Float32, ImmutAnyOrigin], q: Pointer[Float32, ImmutAnyOrigin],
    k: Pointer[Float32, ImmutAnyOrigin], kp: Pointer[Float32, ImmutAnyOrigin],
    xp: Pointer[Float32, ImmutAnyOrigin], adt: Pointer[Float32, ImmutAnyOrigin],
    gamma: Pointer[Float32, ImmutAnyOrigin], beta: Pointer[Float32, ImmutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    var b = Int(block_idx.x) // H
    var h = Int(block_idx.x) % H
    var p = Int(thread_idx.x)
    var S = stack_allocation[N, Float32]()
    for n in range(N):
        S[n] = 0.0
    for t in range(T):
        var base = (b * T + t) * H + h
        var a = exp(adt[unsafe_offset=base])
        var g = gamma[unsafe_offset=base] * x[unsafe_offset=base * P + p]
        var bt = beta[unsafe_offset=base] * xp[unsafe_offset=base * P + p]
        var yv: Float32 = 0.0
        for n in range(N):
            S[n] = a * S[n] + g * k[unsafe_offset=base * N + n] + bt * kp[unsafe_offset=base * N + n]
            yv += S[n] * q[unsafe_offset=base * N + n]
        y[unsafe_offset=base * P + p] = yv


def main() raises:
    var x = List[Float32](length=NX, fill=0.0)
    var q = List[Float32](length=NS, fill=0.0)
    var k = List[Float32](length=NS, fill=0.0)
    var kp = List[Float32](length=NS, fill=0.0)
    var xp = List[Float32](length=NX, fill=0.0)
    var adt = List[Float32](length=NH, fill=0.0)
    var gamma = List[Float32](length=NH, fill=0.0)
    var beta = List[Float32](length=NH, fill=0.0)
    make_inputs(x, q, k, kp, xp, adt, gamma, beta)

    var y_cpu = List[Float32](length=NX, fill=0.0)
    ssd_fwd_cpu(x, q, k, kp, xp, adt, gamma, beta, y_cpu)

    var checksum: Float64 = 0.0
    for i in range(NX):
        checksum += Float64(y_cpu[i])
    print("cpu checksum:", checksum)

    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        with DeviceContext() as ctx:
            var d_x = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_xp = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_q = ctx.enqueue_create_buffer[DType.float32](NS)
            var d_k = ctx.enqueue_create_buffer[DType.float32](NS)
            var d_kp = ctx.enqueue_create_buffer[DType.float32](NS)
            var d_adt = ctx.enqueue_create_buffer[DType.float32](NH)
            var d_gamma = ctx.enqueue_create_buffer[DType.float32](NH)
            var d_beta = ctx.enqueue_create_buffer[DType.float32](NH)
            var d_y = ctx.enqueue_create_buffer[DType.float32](NX)
            d_x.enqueue_copy_from(x.unsafe_ptr())
            d_xp.enqueue_copy_from(xp.unsafe_ptr())
            d_q.enqueue_copy_from(q.unsafe_ptr())
            d_k.enqueue_copy_from(k.unsafe_ptr())
            d_kp.enqueue_copy_from(kp.unsafe_ptr())
            d_adt.enqueue_copy_from(adt.unsafe_ptr())
            d_gamma.enqueue_copy_from(gamma.unsafe_ptr())
            d_beta.enqueue_copy_from(beta.unsafe_ptr())
            ctx.enqueue_function[ssd_fwd_gpu](
                d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_y,
                grid_dim=B * H, block_dim=P,
            )
            ctx.synchronize()
            var y_gpu = List[Float32](length=NX, fill=0.0)
            d_y.enqueue_copy_to(y_gpu.unsafe_ptr())
            ctx.synchronize()
            var max_diff: Float32 = 0.0
            var gpu_sum: Float64 = 0.0
            for i in range(NX):
                var d = y_gpu[i] - y_cpu[i]
                if d < 0:
                    d = -d
                if d > max_diff:
                    max_diff = d
                gpu_sum += Float64(y_gpu[i])
            print("gpu checksum:", gpu_sum)
            print("max |gpu - cpu|:", max_diff)
            var iters = 10
            var t0 = perf_counter_ns()
            for _ in range(iters):
                ctx.enqueue_function[ssd_fwd_gpu](
                    d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_y,
                    grid_dim=B * H, block_dim=P,
                )
            ctx.synchronize()
            print("gpu ms/call:", Float64(perf_counter_ns() - t0) / 1e6 / Float64(iters))


