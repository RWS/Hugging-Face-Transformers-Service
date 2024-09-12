from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os
import sys

# Get the package path
package_path = get_package_paths('llama_cpp')[0]

# Collect data files
datas = collect_data_files('llama_cpp')

# Append DLL based on OS
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'lib', 'llama.dll')  # Adjusted path here
    if os.path.exists(dll_path):
        datas.append((dll_path, 'llama_cpp'))  # The destination will be in the 'llama_cpp' folder
elif sys.platform == 'darwin':  # Mac
    so_path = os.path.join(package_path, 'lib', 'llama.dylib')
    if os.path.exists(so_path):
        datas.append((so_path, 'llama_cpp'))
elif os.name == 'posix':  # Linux
    so_path = os.path.join(package_path, 'lib', 'libllama.so')
    if os.path.exists(so_path):
        datas.append((so_path, 'llama_cpp'))