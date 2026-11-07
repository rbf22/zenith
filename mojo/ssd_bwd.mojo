from std.math import exp, sin
from std.memory import stack_allocation
from layout import TileTensor, row_major
from layout import stack_allocation as shm_alloc
from max.gpu.primitives.block import sum as bsum
from max.gpu.sync import barrier as gbbarrier
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace

comptime P = 64
comptime N = 128
comptime B = 1
comptime T = 48
comptime H = 2
comptime NX = B * T * H * P
comptime NS = B * T * H * N
comptime NH = B * T * H


def gen(n: Int, salt: Int) -> Float64:
    return sin(Float64(n * 131 + salt * 7919) * 0.001)


def make_inputs(
    mut x: List[Float32], mut q: List[Float32], mut k: List[Float32], mut kp: List[Float32],
    mut xp: List[Float32], mut adt: List[Float32], mut gamma: List[Float32], mut beta: List[Float32],
    mut dy: List[Float32],
):
    for i in range(NX):
        x[i] = Float32(gen(i, 1))
        xp[i] = Float32(gen(i, 2))
        dy[i] = Float32(gen(i, 9))
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
    adt: List[Float32], gamma: List[Float32], beta: List[Float32],
    mut y: List[Float32], mut states: List[Float32],
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
                    states[base * P * N + p * N + n] = S[n]
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




comptime NST = B * T * H * P * N


def ssd_bwd_cpu(
    x: List[Float32], q: List[Float32], k: List[Float32], kp: List[Float32], xp: List[Float32],
    adt: List[Float32], gamma: List[Float32], beta: List[Float32], dy: List[Float32],
    states: List[Float32],
    mut dx: List[Float32], mut dxp: List[Float32], mut dq: List[Float32], mut dk: List[Float32],
    mut dkp: List[Float32], mut dadt: List[Float32], mut dgamma: List[Float32], mut dbeta: List[Float32],
):
    var dg = List[Float32](length=NH, fill=0.0)
    var db = List[Float32](length=NH, fill=0.0)
    var da = List[Float32](length=NH, fill=0.0)
    for bh in range(B * H):
        var b = bh // H
        var h = bh % H
        for p in range(P):
            var A = List[Float32](length=N, fill=0.0)
            for t in range(T - 1, -1, -1):
                var base = (b * T + t) * H + h
                var dyv = dy[base * P + p]
                var a_next: Float32 = 0.0
                if t < T - 1:
                    a_next = exp(adt[(b * T + t + 1) * H + h])
                var dot_k: Float32 = 0.0
                var dot_kp: Float32 = 0.0
                var dot_sp: Float32 = 0.0
                for n in range(N):
                    A[n] = dyv * q[base * N + n] + a_next * A[n]
                    dot_k += A[n] * k[base * N + n]
                    dot_kp += A[n] * kp[base * N + n]
                    if t > 0:
                        dot_sp += A[n] * states[(base - H) * P * N + p * N + n]
                var gt = gamma[base]
                var bt = beta[base]
                var xt = x[base * P + p]
                var xpt = xp[base * P + p]
                dx[base * P + p] = gt * dot_k
                dxp[base * P + p] = bt * dot_kp
                dg[base] += xt * dot_k
                db[base] += xpt * dot_kp
                da[base] += dot_sp
                for n in range(N):
                    dk[base * N + n] += gt * xt * A[n]
                    dkp[base * N + n] += bt * xpt * A[n]
                    dq[base * N + n] += dyv * states[base * P * N + p * N + n]
    for base in range(NH):
        dgamma[base] = dg[base]
        dbeta[base] = db[base]
        dadt[base] = exp(adt[base]) * da[base]


def ssd_fwd_store_gpu(
    x: Pointer[Float32, ImmutAnyOrigin], q: Pointer[Float32, ImmutAnyOrigin],
    k: Pointer[Float32, ImmutAnyOrigin], kp: Pointer[Float32, ImmutAnyOrigin],
    xp: Pointer[Float32, ImmutAnyOrigin], adt: Pointer[Float32, ImmutAnyOrigin],
    gamma: Pointer[Float32, ImmutAnyOrigin], beta: Pointer[Float32, ImmutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin], states: Pointer[Float32, MutAnyOrigin],
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
            states[unsafe_offset=base * P * N + p * N + n] = S[n]
            yv += S[n] * q[unsafe_offset=base * N + n]
        y[unsafe_offset=base * P + p] = yv

comptime CH = 32
comptime TILE_LAY = row_major[3, P, CH]()
comptime RED_LAY = row_major[3, P]()


def ssd_bwd_gpu(
    x: Pointer[Float32, ImmutAnyOrigin], q: Pointer[Float32, ImmutAnyOrigin],
    k: Pointer[Float32, ImmutAnyOrigin], kp: Pointer[Float32, ImmutAnyOrigin],
    xp: Pointer[Float32, ImmutAnyOrigin], adt: Pointer[Float32, ImmutAnyOrigin],
    gamma: Pointer[Float32, ImmutAnyOrigin], beta: Pointer[Float32, ImmutAnyOrigin],
    dy: Pointer[Float32, ImmutAnyOrigin], states: Pointer[Float32, ImmutAnyOrigin],
    dx: Pointer[Float32, MutAnyOrigin], dxp: Pointer[Float32, MutAnyOrigin],
    dq: Pointer[Float32, MutAnyOrigin], dk: Pointer[Float32, MutAnyOrigin],
    dkp: Pointer[Float32, MutAnyOrigin], dadt: Pointer[Float32, MutAnyOrigin],
    dgamma: Pointer[Float32, MutAnyOrigin], dbeta: Pointer[Float32, MutAnyOrigin],
):
    var b = Int(block_idx.x) // H
    var h = Int(block_idx.x) % H
    var p = Int(thread_idx.x)
    var A = stack_allocation[N, Float32]()
    var tile = shm_alloc[DType.float32, address_space=.SHARED](TILE_LAY)
    var red = shm_alloc[DType.float32, address_space=.SHARED](RED_LAY)
    for n in range(N):
        A[n] = 0.0
    for t in range(T - 1, -1, -1):
        barrier()
        var base = (b * T + t) * H + h
        var dyv = dy[unsafe_offset=base * P + p]
        var a_next: Float32 = 0.0
        if t < T - 1:
            a_next = exp(adt[unsafe_offset=(b * T + t + 1) * H + h])
        var dot_k: Float32 = 0.0
        var dot_kp: Float32 = 0.0
        var dot_sp: Float32 = 0.0
        for n in range(N):
            A[n] = dyv * q[unsafe_offset=base * N + n] + a_next * A[n]
            dot_k += A[n] * k[unsafe_offset=base * N + n]
            dot_kp += A[n] * kp[unsafe_offset=base * N + n]
            if t > 0:
                dot_sp += A[n] * states[unsafe_offset=(base - H) * P * N + p * N + n]
        var gt = gamma[unsafe_offset=base]
        var bt = beta[unsafe_offset=base]
        var xt = x[unsafe_offset=base * P + p]
        var xpt = xp[unsafe_offset=base * P + p]
        dx[unsafe_offset=base * P + p] = gt * dot_k
        dxp[unsafe_offset=base * P + p] = bt * dot_kp
        for n0 in range(0, N, CH):
            for j in range(CH):
                var n = n0 + j
                tile[0, p, j] = gt * xt * A[n]
                tile[1, p, j] = bt * xpt * A[n]
                tile[2, p, j] = dyv * states[unsafe_offset=base * P * N + p * N + n]
            barrier()
            for j in range(CH):
                var n = n0 + j
                if p < CH:
                    var s0: Float32 = 0.0
                    var s1: Float32 = 0.0
                    var s2: Float32 = 0.0
                    for u in range(P):
                        s0 += tile[0, u, j]
                        s1 += tile[1, u, j]
                        s2 += tile[2, u, j]
                    dk[unsafe_offset=base * N + n] = s0
                    dkp[unsafe_offset=base * N + n] = s1
                    dq[unsafe_offset=base * N + n] = s2
            barrier()
        red[0, p] = xt * dot_k
        red[1, p] = xpt * dot_kp
        red[2, p] = dot_sp
        barrier()
        if p == 0:
            var s0: Float32 = 0.0
            var s1: Float32 = 0.0
            var s2: Float32 = 0.0
            for u in range(P):
                s0 += red[0, u]
                s1 += red[1, u]
                s2 += red[2, u]
            dgamma[unsafe_offset=base] = s0
            dbeta[unsafe_offset=base] = s1
            dadt[unsafe_offset=base] = exp(adt[unsafe_offset=base]) * s2

def report(name: String, gpu_l: List[Float32], cpu_l: List[Float32]):
    var mx: Float32 = 0.0
    for j in range(len(cpu_l)):
        var dd = gpu_l[j] - cpu_l[j]
        if dd < 0:
            dd = -dd
        if dd > mx:
            mx = dd
    print(name, "max |gpu-cpu|:", mx)


def main() raises:
    var x = List[Float32](length=NX, fill=0.0)
    var q = List[Float32](length=NS, fill=0.0)
    var k = List[Float32](length=NS, fill=0.0)
    var kp = List[Float32](length=NS, fill=0.0)
    var xp = List[Float32](length=NX, fill=0.0)
    var adt = List[Float32](length=NH, fill=0.0)
    var gamma = List[Float32](length=NH, fill=0.0)
    var beta = List[Float32](length=NH, fill=0.0)
    var dy = List[Float32](length=NX, fill=0.0)
    make_inputs(x, q, k, kp, xp, adt, gamma, beta, dy)

    var y_cpu = List[Float32](length=NX, fill=0.0)
    var states_cpu = List[Float32](length=NST, fill=0.0)
    ssd_fwd_cpu(x, q, k, kp, xp, adt, gamma, beta, y_cpu, states_cpu)

    var dx_cpu = List[Float32](length=NX, fill=0.0)
    var dxp_cpu = List[Float32](length=NX, fill=0.0)
    var dq_cpu = List[Float32](length=NS, fill=0.0)
    var dk_cpu = List[Float32](length=NS, fill=0.0)
    var dkp_cpu = List[Float32](length=NS, fill=0.0)
    var dadt_cpu = List[Float32](length=NH, fill=0.0)
    var dgamma_cpu = List[Float32](length=NH, fill=0.0)
    var dbeta_cpu = List[Float32](length=NH, fill=0.0)
    ssd_bwd_cpu(x, q, k, kp, xp, adt, gamma, beta, dy, states_cpu,
                dx_cpu, dxp_cpu, dq_cpu, dk_cpu, dkp_cpu, dadt_cpu, dgamma_cpu, dbeta_cpu)

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
            var d_dy = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_y = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_states = ctx.enqueue_create_buffer[DType.float32](NST)
            var d_dx = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_dxp = ctx.enqueue_create_buffer[DType.float32](NX)
            var d_dq = ctx.enqueue_create_buffer[DType.float32](NS)
            var d_dk = ctx.enqueue_create_buffer[DType.float32](NS)
            var d_dkp = ctx.enqueue_create_buffer[DType.float32](NS)
            var d_dadt = ctx.enqueue_create_buffer[DType.float32](NH)
            var d_dgamma = ctx.enqueue_create_buffer[DType.float32](NH)
            var d_dbeta = ctx.enqueue_create_buffer[DType.float32](NH)
            d_x.enqueue_copy_from(x.unsafe_ptr())
            d_xp.enqueue_copy_from(xp.unsafe_ptr())
            d_q.enqueue_copy_from(q.unsafe_ptr())
            d_k.enqueue_copy_from(k.unsafe_ptr())
            d_kp.enqueue_copy_from(kp.unsafe_ptr())
            d_adt.enqueue_copy_from(adt.unsafe_ptr())
            d_gamma.enqueue_copy_from(gamma.unsafe_ptr())
            d_beta.enqueue_copy_from(beta.unsafe_ptr())
            d_dy.enqueue_copy_from(dy.unsafe_ptr())
            ctx.enqueue_function[ssd_fwd_store_gpu](
                d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_y, d_states,
                grid_dim=B * H, block_dim=P,
            )
            ctx.enqueue_function[ssd_bwd_gpu](
                d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_dy, d_states,
                d_dx, d_dxp, d_dq, d_dk, d_dkp, d_dadt, d_dgamma, d_dbeta,
                grid_dim=B * H, block_dim=P,
            )
            ctx.synchronize()
            var out_dx = List[Float32](length=NX, fill=0.0)
            var out_dxp = List[Float32](length=NX, fill=0.0)
            var out_dq = List[Float32](length=NS, fill=0.0)
            var out_dk = List[Float32](length=NS, fill=0.0)
            var out_dkp = List[Float32](length=NS, fill=0.0)
            var out_dadt = List[Float32](length=NH, fill=0.0)
            var out_dgamma = List[Float32](length=NH, fill=0.0)
            var out_dbeta = List[Float32](length=NH, fill=0.0)
            d_dx.enqueue_copy_to(out_dx.unsafe_ptr())
            d_dxp.enqueue_copy_to(out_dxp.unsafe_ptr())
            d_dq.enqueue_copy_to(out_dq.unsafe_ptr())
            d_dk.enqueue_copy_to(out_dk.unsafe_ptr())
            d_dkp.enqueue_copy_to(out_dkp.unsafe_ptr())
            d_dadt.enqueue_copy_to(out_dadt.unsafe_ptr())
            d_dgamma.enqueue_copy_to(out_dgamma.unsafe_ptr())
            d_dbeta.enqueue_copy_to(out_dbeta.unsafe_ptr())
            ctx.synchronize()
            report("dx", out_dx, dx_cpu)
            report("dxp", out_dxp, dxp_cpu)
            report("dq", out_dq, dq_cpu)
            report("dk", out_dk, dk_cpu)
            report("dkp", out_dkp, dkp_cpu)
            report("dadt", out_dadt, dadt_cpu)
            report("dgamma", out_dgamma, dgamma_cpu)
            report("dbeta", out_dbeta, dbeta_cpu)
            var iters = 10
            var t0 = perf_counter_ns()
            for _ in range(iters):
                ctx.enqueue_function[ssd_fwd_store_gpu](
                    d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_y, d_states,
                    grid_dim=B * H, block_dim=P,
                )
                ctx.enqueue_function[ssd_bwd_gpu](
                    d_x, d_q, d_k, d_kp, d_xp, d_adt, d_gamma, d_beta, d_dy, d_states,
                    d_dx, d_dxp, d_dq, d_dk, d_dkp, d_dadt, d_dgamma, d_dbeta,
                    grid_dim=B * H, block_dim=P,
                )
            ctx.synchronize()
            print("gpu fwd+bwd ms/pair:", Float64(perf_counter_ns() - t0) / 1e6 / Float64(iters))
