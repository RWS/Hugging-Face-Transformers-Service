from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, get_package_paths
import os
import logging

# Initialize logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Collect dynamic libraries using PyInstaller's utility
binaries = collect_dynamic_libs('llama_cpp')
logger.debug(f'Initially collected binaries from llama_cpp: {binaries}')

# Get the package paths for llama_cpp
package_paths = get_package_paths('llama_cpp')
if not package_paths:
    logger.error('Could not find package path for llama_cpp.')
    package_path = ''
else:
    package_path = package_paths[0]
    logger.debug(f'Package path for llama_cpp: {package_path}')

    # Define the 'lib' directory within llama_cpp
    lib_dir = os.path.join(package_path, 'lib')

    # Check if the 'lib' directory exists
    if os.path.isdir(lib_dir):
        # Iterate over all DLL files in the 'lib' directory
        for filename in os.listdir(lib_dir):
            if filename.lower().endswith('.dll'):
                dll_path = os.path.join(lib_dir, filename)
                if os.path.isfile(dll_path):
                    # Destination directory within the bundled app
                    dest_dir = os.path.join('llama_cpp', 'lib')
                    binaries.append((dll_path, dest_dir))
                    logger.debug(f'Added DLL to binaries: {dll_path} -> {dest_dir}')
                else:
                    logger.warning(f'Path is not a file, skipping: {dll_path}')
    else:
        logger.warning(f'Lib directory does not exist: {lib_dir}')

# Collect data files (if any)
datas = collect_data_files('llama_cpp')
logger.debug(f'Collected data files from llama_cpp: {datas}')