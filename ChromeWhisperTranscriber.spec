# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

def _collect_package_assets(package):
    try:
        hidden = collect_submodules(package)
    except ModuleNotFoundError:
        hidden = []
    try:
        binaries = collect_dynamic_libs(package)
    except ModuleNotFoundError:
        binaries = []
    try:
        datas = collect_data_files(package)
    except ModuleNotFoundError:
        datas = []
    return hidden, binaries, datas

faster_hidden, faster_bins, faster_datas = _collect_package_assets('faster_whisper')
ctrans_hidden, ctrans_bins, ctrans_datas = _collect_package_assets('ctranslate2')
soundcard_hidden, soundcard_bins, soundcard_datas = _collect_package_assets('soundcard')


def _dedupe(seq):
    seen = {}
    for item in seq:
        if item not in seen:
            seen[item] = None
    return list(seen.keys())

extra_hidden = _dedupe(['faster_whisper', 'ctranslate2'] + faster_hidden + ctrans_hidden + soundcard_hidden)
extra_binaries = faster_bins + ctrans_bins + soundcard_bins
extra_datas = faster_datas + ctrans_datas + soundcard_datas

a = Analysis(
    ['chrome_whisper_transcriber_local.py'],
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
    name='ChromeWhisperTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='ChromeWhisperTranscriber',
)
