import ctypes
lib=ctypes.CDLL(r'D:\infinitensor\tuili\hw3\llaisys\python\llaisys\llaisys.dll')
print('has create attr:', hasattr(lib, 'llaisysQwen2ModelCreate'))
try:
    f = lib.llaisysQwen2ModelCreate
    print('got create:', f)
except Exception as e:
    print('error getting create:', e)

print('has tensorCreate:', hasattr(lib, 'tensorCreate'))
try:
    f2 = lib.tensorCreate
    print('got tensorCreate:', f2)
except Exception as e:
    print('error getting tensorCreate:', e)
