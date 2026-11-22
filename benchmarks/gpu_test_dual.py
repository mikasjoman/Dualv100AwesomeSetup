import ctypes
import time
import sys
import subprocess

# OpenCL definitions
cl = ctypes.CDLL('libOpenCL.so.1')

cl.clCreateBuffer.restype = ctypes.c_void_p
cl.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]

cl.clEnqueueNDRangeKernel.restype = ctypes.c_int32
cl.clEnqueueNDRangeKernel.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_void_p
]

def check(err, msg):
    if err != 0:
        raise RuntimeError(f"{msg}: error code {err}")

platform = ctypes.c_void_p()
check(cl.clGetPlatformIDs(1, ctypes.byref(platform), None), "clGetPlatformIDs")

num_devs = ctypes.c_uint32()
check(cl.clGetDeviceIDs(platform, 0xFFFFFFFF, 0, None, ctypes.byref(num_devs)), "clGetDeviceIDs count")
devices = (ctypes.c_void_p * num_devs.value)()
check(cl.clGetDeviceIDs(platform, 0xFFFFFFFF, num_devs.value, devices, None), "clGetDeviceIDs list")

# Target device 0 and device 1
target_indices = [0, 1]
target_devices = (ctypes.c_void_p * 2)(devices[0], devices[1])

for idx in target_indices:
    name = ctypes.create_string_buffer(256)
    cl.clGetDeviceInfo(devices[idx], 0x102B, 256, name, None)
    print(f"[INIT] Device {idx}: {name.value.decode('utf-8', errors='ignore')}")

err = ctypes.c_int32()
ctx = cl.clCreateContext(None, 2, target_devices, None, None, ctypes.byref(err))
check(err.value, "clCreateContext")

kernel_src = b"""
__kernel void stress(__global float* buf, int intensity) {
    int gid = get_global_id(0);
    float a = (float)gid * 0.001f;
    float b = 1.0001f;
    float c = 0.5f;
    for (int i = 0; i < intensity; i++) {
        a = mad(a, b, c);
        b = mad(b, c, a);
        c = mad(c, a, b);
    }
    buf[gid] = a + b + c;
}
"""

src_ptr = ctypes.c_char_p(kernel_src)
src_len = ctypes.c_size_t(len(kernel_src))
prog = cl.clCreateProgramWithSource(ctx, 1, ctypes.byref(src_ptr), ctypes.byref(src_len), ctypes.byref(err))
check(err.value, "clCreateProgramWithSource")

build_err = cl.clBuildProgram(prog, 2, target_devices, None, None, None)
if build_err != 0:
    for idx in target_indices:
        log = ctypes.create_string_buffer(4096)
        cl.clGetProgramBuildInfo(prog, devices[idx], 0x1183, 4096, log, None)
        print(f"Build log dev {idx}:", log.value.decode())
    sys.exit(1)

queues = []
kernels = []
buffers = []

for idx in target_indices:
    q = cl.clCreateCommandQueue(ctx, devices[idx], 0, ctypes.byref(err))
    check(err.value, f"clCreateCommandQueue dev {idx}")
    queues.append(q)

    k = cl.clCreateKernel(prog, b"stress", ctypes.byref(err))
    check(err.value, f"clCreateKernel dev {idx}")
    kernels.append(k)

    buf_size = 64 * 1024 * 1024
    d_buf = cl.clCreateBuffer(ctx, 1, buf_size, None, ctypes.byref(err))
    check(err.value, f"clCreateBuffer dev {idx}")
    buf_mem = ctypes.c_void_p(d_buf)
    cl.clSetKernelArg(k, 0, ctypes.c_size_t(ctypes.sizeof(ctypes.c_void_p)), ctypes.byref(buf_mem))

    intensity = ctypes.c_int32(5000)
    cl.clSetKernelArg(k, 1, ctypes.c_size_t(ctypes.sizeof(ctypes.c_int32)), ctypes.byref(intensity))
    buffers.append(d_buf)

def get_telemetry():
    res = subprocess.check_output(
        ["nvidia-smi", "-i", "0,1", "--query-gpu=index,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        text=True
    ).strip().splitlines()
    data = {}
    for line in res:
        parts = [p.strip() for p in line.split(",")]
        data[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return data

global_work_size = ctypes.c_size_t(2 * 1024 * 1024)
local_work_size = ctypes.c_size_t(256)

SAFETY_TEMP_LIMIT = 77.0
DURATION_SEC = 180  # 3 minuter test

print(f"=== STARTAR DUAL TESLA V100 LOAD TEST (Max Limit: 160W, Säkerhetsgräns: {SAFETY_TEMP_LIMIT}°C) ===")
sys.stdout.flush()

start_time = time.time()
last_report = 0

try:
    while time.time() - start_time < DURATION_SEC:
        for q, k in zip(queues, kernels):
            cl.clEnqueueNDRangeKernel(q, k, 1, None, ctypes.byref(global_work_size), ctypes.byref(local_work_size), 0, None, None)
            cl.clFlush(q)
        time.sleep(0.0001)

        now = time.time()
        if now - last_report >= 3.0:
            telem = get_telemetry()
            elapsed = int(now - start_time)
            t0, p0 = telem.get(0, (0.0, 0.0))
            t1, p1 = telem.get(1, (0.0, 0.0))
            print(f"[{elapsed:3d}s] GPU 0: {t0:.1f}°C, {p0:.1f}W | GPU 1: {t1:.1f}°C, {p1:.1f}W")
            sys.stdout.flush()
            last_report = now

            if t0 >= SAFETY_TEMP_LIMIT or t1 >= SAFETY_TEMP_LIMIT:
                hot_gpu = 0 if t0 >= SAFETY_TEMP_LIMIT else 1
                hot_temp = max(t0, t1)
                print(f"\n[NÖDSTOPP] GPU {hot_gpu} nådde {hot_temp:.1f}°C (>= {SAFETY_TEMP_LIMIT}°C)! Avbryter omedelbart!")
                sys.stdout.flush()
                break

except KeyboardInterrupt:
    print("\n[AVBRUTET] Användaravbrott.")

finally:
    for q in queues:
        cl.clFinish(q)
    final_telem = get_telemetry()
    t0, p0 = final_telem.get(0, (0.0, 0.0))
    t1, p1 = final_telem.get(1, (0.0, 0.0))
    print(f"\n=== Test slutfört! GPU 0: {t0:.1f}°C ({p0:.1f}W) | GPU 1: {t1:.1f}°C ({p1:.1f}W) ===")
    sys.stdout.flush()
