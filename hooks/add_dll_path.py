import sys
import os
import logging

# Initialize logging
logging.basicConfig(
   filename=os.path.join(os.path.dirname(sys.executable), 'runtime_hook.log'),
   filemode='w',
   level=logging.DEBUG,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def adjust_path():
    if getattr(sys, 'frozen', False):
	   # In --onedir mode, dependencies are inside '_internal'
        base_path = os.path.dirname(sys.executable)
        logger.debug(f'Runtime Hook: Running in frozen mode. Base path: {base_path}')
        llama_cpp_lib = os.path.join(base_path, '_internal', 'llama_cpp', 'lib')
    else:
	   # Running in normal Python environment
        base_path = os.path.abspath(".")
        logger.debug(f'Runtime Hook: Running in normal mode. Base path: {base_path}')
        llama_cpp_lib = os.path.join(base_path, 'llama_cpp', 'lib')

    logger.debug(f'Runtime Hook: Adding to PATH: {llama_cpp_lib}')
    os.environ['PATH'] = llama_cpp_lib + os.pathsep + os.environ.get('PATH', '')
    logger.debug(f'Runtime Hook: Updated PATH: {os.environ["PATH"]}')

    # Verify DLLs presence
    required_dlls = [
	   'llama.dll',
	   'ggml.dll',
	   'ggml-amx.dll',
	   'ggml-base.dll',
	   'ggml-cpu.dll',
	   'llava.dll'
   ]
    for dll in required_dlls:
        dll_path = os.path.join(llama_cpp_lib, dll)
        if os.path.exists(dll_path):
            logger.debug(f'Runtime Hook: Found DLL: {dll_path}')
        else:
            logger.error(f'Runtime Hook: Missing DLL: {dll_path}')

# Execute the path adjustment
adjust_path()