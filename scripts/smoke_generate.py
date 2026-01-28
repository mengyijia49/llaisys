import ctypes
from ctypes import c_size_t, c_int, c_int64, c_void_p, c_char_p, POINTER
import numpy as np
import sys

DLL = r'D:\infinitensor\tuili\hw3\llaisys\python\llaisys\llaisys.dll'
print('Loading DLL', DLL)
lib = ctypes.CDLL(DLL)

# prototypes
lib.tensorCreate.argtypes = [POINTER(c_size_t), c_size_t, c_int, c_int, c_int]
lib.tensorCreate.restype = c_void_p
lib.tensorLoad.argtypes = [c_void_p, c_void_p]
lib.tensorLoad.restype = None
lib.tensorDestroy.argtypes = [c_void_p]
lib.tensorDestroy.restype = None

lib.llaisysQwen2ModelCreate.argtypes = [POINTER(ctypes.c_void_p), c_int, POINTER(c_int), c_int]
# We'll not use this prototype; define a simpler one matching earlier header
# But to be safe use c_void_p for meta pointer by building a struct in Python not trivial; instead use direct Create with bytes

# Define meta struct layout in Python using ctypes
class Meta(ctypes.Structure):
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
        ('epsilon', ctypes.c_float),
        ('theta', ctypes.c_float),
        ('end_token', c_int64),
    ]

lib.llaisysQwen2ModelCreate.argtypes = [POINTER(Meta), c_int, POINTER(c_int), c_int]
lib.llaisysQwen2ModelCreate.restype = c_void_p
lib.llaisysQwen2ModelSetWeight.argtypes = [c_void_p, c_char_p, c_void_p]
lib.llaisysQwen2ModelSetWeight.restype = c_int
lib.llaisysQwen2ModelFinalize.argtypes = [c_void_p]
lib.llaisysQwen2ModelFinalize.restype = c_int
lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
lib.llaisysQwen2ModelInfer.restype = c_int64

# Create small meta
meta = Meta()
meta.dtype = 13  # F32
meta.nlayer = 0  # no layers to avoid accessing per-layer weights
meta.hs = 16
meta.nh = 4
meta.nkvh = 4
meta.dh = 4
meta.di = 64
meta.maxseq = 128
meta.voc = 100
meta.epsilon = 1e-5
meta.theta = 1.0
meta.end_token = -1

print('Creating model')
model = lib.llaisysQwen2ModelCreate(ctypes.byref(meta), 0, None, 0)
if not model:
    print('Model create failed', file=sys.stderr)
    sys.exit(2)
print('Model ptr', model)

# create in_embed tensor shape [voc, hs]
voc = int(meta.voc)
hs = int(meta.hs)
shape = (c_size_t * 2)(voc, hs)
emb_tensor = lib.tensorCreate(shape, 2, 13, 0, 0)
# fill with random floats
arr = (np.random.rand(voc, hs).astype(np.float32)).ctypes
lib.tensorLoad(emb_tensor, arr.data)
print('in_embed created')

# create out_embed tensor shape [voc, hs]
shape2 = (c_size_t * 2)(voc, hs)
out_tensor = lib.tensorCreate(shape2, 2, 13, 0, 0)
arr2 = (np.random.rand(hs, voc).astype(np.float32)).ctypes
lib.tensorLoad(out_tensor, arr2.data)
print('out_embed created')

# set weights
ret = lib.llaisysQwen2ModelSetWeight(model, b'embed_tokens.weight', emb_tensor)
print('set in_embed', ret)
ret = lib.llaisysQwen2ModelSetWeight(model, b'lm_head.weight', out_tensor)
print('set out_embed', ret)

# finalize
lib.llaisysQwen2ModelFinalize(model)
print('finalized')

# infer on token ids [1,2,3]
seq = (c_int64 * 3)(1, 2, 3)
nexttok = lib.llaisysQwen2ModelInfer(model, seq, 3)
print('next token ->', nexttok)

# cleanup
lib.tensorDestroy(emb_tensor)
lib.tensorDestroy(out_tensor)
lib.llaisysQwen2ModelDestroy(ctypes.c_void_p(model))
print('done')
