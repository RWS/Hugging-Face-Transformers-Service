# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
import os

import site
import glob
from pathlib import Path
from PyInstaller.building.build_main import Analysis, PYZ, EXE

# Collect dynamic libraries from 'llama_cpp'
binaries = []

# Collect data files from 'llama_cpp'
datas = []

# Additional data files
additional_datas = [
  ('requirements.txt', '.'), 
  ('LICENSE', '.'), 
  ('README.md', '.'), 
  ('src/api.py', 'src/'), 
  ('src/app.py', 'src/'),
  ('src/config.py', 'src/'), 
  ('src/connection_manager.py', 'src/'), 
  ('src/helpers.py', 'src/'), 
  ('src/main.py', 'src/'), 
  ('src/models.py', 'src/'), 
  ('src/state.py', 'src/')
]

# Define system DLLs to exclude
system_dlls = [
   'Kernel32.dll',
   'Advapi32.dll',
   'VCOMP140.dll',
   'MSVCP140.dll',
   'VCRUNTIME140.dll',
   'VCRUNTIME140_1.dll',
]

a = Analysis(
  ['src/main.py'],              # Main script
  pathex=['.'],                 # Path to main script
  binaries=binaries,            # Binaries handled by hook-llama_cpp.py or manually
  datas=datas + additional_datas,
  hiddenimports=['tzdata'],     # Ensure tzdata is included
  hookspath=['./hooks'],        # Path to custom hooks
  hooksconfig={},
  runtime_hooks=[],
  excludes=[],        
  noarchive=False,
  optimize=0,
)

# filter out system DLLs
def filter_system_dlls(binaries, system_dlls):
   filtered = []
   for binary in binaries:
	   dll_name = os.path.basename(binary[0]).lower()
	   if dll_name not in [dll.lower() for dll in system_dlls]:
		   filtered.append(binary)
   return filtered

a.binaries = filter_system_dlls(a.binaries, system_dlls)
# logger.debug(f'Filtered binaries: {a.binaries}')

pyz = PYZ(a.pure)

exe = EXE(
  pyz,
  a.scripts,
  a.binaries,
  a.datas,
  [],
  name='HuggingFace-TS',
  debug=False,
  bootloader_ignore_signals=False,
  strip=False,
  upx=True,
  upx_exclude=[],
  runtime_tmpdir=None,
  console=True, 
  disable_windowed_traceback=False,
  argv_emulation=False,
  target_arch=None,
  codesign_identity=None,
  entitlements_file=None,
)