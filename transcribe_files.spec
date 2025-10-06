# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

try:
    soundcard_hidden = collect_submodules('soundcard')
    soundcard_bins = collect_dynamic_libs('soundcard')
    soundcard_datas = collect_data_files('soundcard')
except ModuleNotFoundError:
    soundcard_hidden = []
    soundcard_bins = []
    soundcard_datas = []

extra_hidden = ['faster_whisper'] + soundcard_hidden
extra_binaries = soundcard_bins
extra_datas = soundcard_datas

a = Analysis(
    ['transcribe_files.py'],
    pathex=[],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=extra_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='transcribe_files',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='transcribe_files',
)

