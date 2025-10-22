# import os, sys, ctypes, subprocess
# import torch

# print("Python exe:", sys.executable)
# print("Platform:", sys.platform)
# print("torch.__version__:", torch.__version__)
# print("torch.version.cuda:", torch.version.cuda)
# print("torch.cuda.is_available():", torch.cuda.is_available())
# try:
#     print("torch.cuda.device_count():", torch.cuda.device_count())
# except Exception as e:
#     print("device_count error:", e)
# print("torch._C:", getattr(torch._C, "__file__", "no torch._C"))

# # show where pip installed torch files
# try:
#     import pkgutil, torch
#     print("torch location:", torch.__file__)
# except Exception as e:
#     print("torch import error:", e)

# # try to load CUDA runtime DLL (Windows)
# dlls = ["cudart64_12.dll", "cudart64_12_6.dll", "cudart64_11.dll"]
# for d in dlls:
#     try:
#         ctypes.CDLL(d)
#         print("Loaded DLL:", d)
#     except Exception as e:
#         print("Can't load", d, "->", repr(e))

# # print PATH (shortened)
# p = os.environ.get("PATH","").split(os.pathsep)
# print("PATH entries (first 20):")
# for i,entry in enumerate(p[:20], 1):
#     print(i, entry)



# import torch

# # Проверяем, доступен ли GPU (CUDA)
# print("CUDA доступна:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("Имя GPU:", torch.cuda.get_device_name(0))
#     print("Количество GPU:", torch.cuda.device_count())

#     # Простая проверка вычислений на GPU
#     x = torch.rand(3, 3).to("cuda")
#     y = torch.rand(3, 3).to("cuda")
#     z = torch.matmul(x, y)
#     print("Результат умножения на GPU:")
#     print(z)
# else:
#     print("PyTorch использует CPU. GPU не найден или CUDA не установлена.")



import importlib, os, traceback, ctypes
try:
    m = importlib.import_module('torchcodec')
    pkg_dir = os.path.dirname(m.__file__)
    print("torchcodec package:", m.__file__)
except Exception as e:
    # если импорт упал — всё равно попытаемся найти папку
    import pkgutil, sys
    mod = pkgutil.find_loader('torchcodec')
    if mod is None:
        print("torchcodec not importable and not found via pkgutil")
        raise
    pkg_dir = os.path.dirname(mod.get_filename())
    print("torchcodec located at:", pkg_dir)

dll = os.path.join(pkg_dir, "libtorchcodec_core7.dll")
print("Testing DLL:", dll)
try:
    ctypes.WinDLL(dll)
    print("DLL loaded successfully (ctypes.WinDLL).")
except Exception:
    print("ctypes.WinDLL failed, traceback:")
    traceback.print_exc()
