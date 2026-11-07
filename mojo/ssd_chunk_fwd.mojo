from std.math import exp, sin
from std.memory import stack_allocation
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime P = 64
comptime N = 128
comptime B = 2
comptime T = 192
comptime H = 3
comptime L = 64
comptime NC = T // L
comptime NX = B * T * H * P
comptime NS = B * T * H * N
comptime NH = B * T * H
comptime NSLOC = B * H * NC * P * N


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


def chunk_local_gpu(
    x: Pointer[Float32, ImmutAnyOrigin], q: Pointer[Float32, ImmutAnyOrigin],
    k: Pointer[Float32, ImmutAnyOrigin], kp: Pointer[Float32, ImmutAnyOrigin],
    xp: Pointer[Float32, ImmutAnyOrigin], adt: Pointer[Float32, ImmutAnyOrigin],
    gamma: Pointer[Float32, ImmutAnyOrigin], beta: Pointer[Float32, ImmutAnyOrigin],
    yintra: Pointer[Float32, MutAnyOrigin], sloc: Pointer[Float32, MutAnyOrigin],
):
    var bhc = Int(block_idx.x)
    var c = bhc % NC
    var bh = bhc // NC
    var b = bh // H
    var h = bh % H
    var tl = Int(thread_idx.x)
    var t0 = c * L
    var cs = stack_allocation[L, Float32]()
    var run: Float32 = 0.0
    for s in range(L):
        run += adt[unsafe_offset=(b * T + t0 + s) * H + h]
        cs[s] = run
    var yacc = stack_allocation[P, Float32]()
    for p in range(P):
        yacc[p] = 0.0
    var tb = (b * T + t0 + tl) * H + h
    for s in range(tl + 1):
        var sb = (b * T + t0 + s) * H + h
        var d1: Float32 = 0.0
        var d2: Float32 = 0.0
        for n in range(N):
            d1 += q[unsafe_offset=tb * N + n] * k[unsafe_offset=sb * N + n]
            d2 += q[unsafe_offset=tb * N + n] * kp[unsafe_offset=sb * N + n]
        var w = exp(cs[tl] - cs[s])
        var c1 = w * d1 * gamma[unsafe_offset=sb]
        var c2 = w * d2 * beta[unsafe_offset=sb]
        for p in range(P):
            yacc[p] += c1 * x[unsafe_offset=sb * P + p] + c2 * xp[unsafe_offset=sb * P + p]
    for p in range(P):
        yintra[unsafe_offset=tb * P + p] = yacc[p]
    var last = cs[L - 1]
    var wg = stack_allocation[L, Float32]()
    var wb = stack_allocation[L, Float32]()
    for s in range(L):
        var sb = (b * T + t0 + s) * H + h
        var w = exp(last - cs[s])
        wg[s] = w * gamma[unsafe_offset=sb]
        wb[s] = w * beta[unsafe_offset=sb]
    for idx in range(tl, P * N, L):
        var p = idx // N
        var n = idx % N
        var v: Float32 = 0.0
        for s in range(L):
            var sb = (b * T + t0 + s) * H + h
            v += wg[s] * x[unsafe_offset=sb * P + p] * k[unsafe_offset=sb * N + n] + wb[s] * xp[unsafe_offset=sb * P + p] * kp[unsafe_offset=sb * N + n]
        sloc[bhc * P * N + idx] = v


def chunk_carry_gpu(
    q: Pointer[Float32, ImmutAnyOrigin], adt: Pointer[Float32, ImmutAnyOrigin],
    sloc: Pointer[Float32, ImmutAnyOrigin], yintra: Pointer[Float32, ImmutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    var bhc = Int(block_idx.x)
    var c = bhc % NC
    var bh = bhc // NC
    var b = bh // H
    var h = bh % H
    var p = Int(thread_idx.x)
    var S = stack_allocation[N, Float32]()
    for n in range(N):
        S[n] = 0.0
    for j in range(c):
        var cend: Float32 = 0.0
        for s in range(L):
            cend += adt[unsafe_offset=(b * T + j * L + s) * H + h]
        var a = exp(cend)
        for n in range(N):
            S[n] = a * S[n] + sloc[(bh * NC + j) * P * N + p * N + n]
    var run: Float32 = 0.0
    for tl in range(L):
        var t = c * L + tl
        var tb = (b * T + t) * H + h
        run += adt[unsafe_offset=tb]
        var acc: Float32 = 0.0
        for n in range(N):
            acc += S[n] * q[unsafe_offset=tb * N + n]
        y[unsafe_offset=tb * P + p] = yintra[unsafe_offset=tb * P + p] + exp(run) * acc


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
            var d_yintra = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_sloc = ctx.enqueue_create_buffer[DType.float32](NSLOC)
            var d_y = ctx.enqueue_create_buffer[DType.float32](NX)
            d_x.enqueue_copy_from(x.unsafe_ptr())
            d_xp.enqueue_copy_from(xp.unsafe_ptr())
            d_q.enqueue_copy_from(q.unsafe_ptr())
            d_k.enqueue_copy_from(k.unsafe_ptr())
            d_kp.enqueue_copy_from(kp.unsafe_ptr())
            d_adt.enqueue_copy_from(adt.unsafe_ptr())
            d_gamma.enqueue_copy_from(gamma.unsafe_ptr())
            d_beta.enqueue_copy_from(beta.unsafe_ptr())
            ctx.enqueue_function[chunk_local_gpu](
                d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_yintra, d_sloc,
                grid_dim=B * H * NC, block_dim=L,
            )
            ctx.enqueue_function[chunk_carry_gpu](
                d_q, d_adt, d_sloc, d_yintra, d_y,
                grid_dim=B * H * NC, block_dim=P,
            )
            ctx.synchronize()
            var y_gpu = List[Float32](length=NX, fill=0.0)
            d_y.enqueue_copy_to(y_gpu.unsafe_ptr())
            ctx.synchronize()
            var mx: Float32 = 0.0
            for i in range(NX):
                var dd = y_gpu[i] - y_cpu[i]
                if dd < 0:
                    dd = -dd
                if dd > mx:
                    mx = dd
            print("max |chunked_gpu - cpu|:", mx)
            var iters = 10
            var t0 = perf_counter_ns()
            for _ in range(iters):
                ctx.enqueue_function[chunk_local_gpu](
                    d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_yintra, d_sloc,
                    grid_dim=B * H * NC, block_dim=L,
                )
                ctx.enqueue_function[chunk_carry_gpu](
                    d_q, d_adt, d_sloc, d_yintra, d_y,
                    grid_dim=B * H * NC, block_dim=P,
                )
            ctx.synchronize()
            print("chunked gpu fwd ms/call:", Float64(perf_counter_ns() - t0) / 1e6 / Float64(iters))
