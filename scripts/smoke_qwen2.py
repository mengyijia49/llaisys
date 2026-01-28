import ctypes
from ctypes import Structure, POINTER, c_size_t, c_int, c_int64, c_float, c_void_p
import sys

DLL = r'D:\infinitensor\tuili\hw3\llaisys\python\llaisys\llaisys.dll'
print('Loading DLL:', DLL)
lib = ctypes.CDLL(DLL)

class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ('dtype', c_int),
        ('nlayer', c_size_t),
        ('hs', c_size_t),
        ('nh', c_size_t),
        ('nkvh', c_size_t),
        ('dh', c_size_t),
        ('di', c_size_t),
        ('maxseq', c_size_t),
        ('voc', c_size_t),
        ('epsilon', c_float),
        ('theta', c_float),
        ('end_token', c_int64),
    ]

# prototypes
lib.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), c_int, POINTER(c_int), c_int]
lib.llaisysQwen2ModelCreate.restype = c_void_p
lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
lib.llaisysQwen2ModelDestroy.restype = None

m = LlaisysQwen2Meta()
m.dtype = 13  # LLAISYS_DTYPE_F32
m.nlayer = 1
m.hs = 16
m.nh = 4
m.nkvh = 4
m.dh = 4
m.di = 64
m.maxseq = 128
m.voc = 1000
m.epsilon = 1e-5
m.theta = 1.0
m.end_token = -1

print('Calling ll_create...')
model = lib.llaisysQwen2ModelCreate(ctypes.byref(m), 0, None, 0)
print('ll_create returned:', model)
if not model:
    print('create failed', file=sys.stderr)
    sys.exit(2)

print('Calling ll_destroy...')
lib.llaisysQwen2ModelDestroy(model)
print('destroy ok')
