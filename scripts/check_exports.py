import ctypes
lib=ctypes.CDLL(r'D:\infinitensor\tuili\hw3\llaisys\python\llaisys\llaisys.dll')
print('handle=', lib._handle)
GetProcAddress=ctypes.windll.kernel32.GetProcAddress
print('create=', GetProcAddress(lib._handle, b'llaisysQwen2ModelCreate'))
print('tensor=', GetProcAddress(lib._handle, b'tensorCreate'))
print('done')
