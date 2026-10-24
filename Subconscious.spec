# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('src/subconscious', 'subconscious'), ('LICENCE', '.'), ('THIRD_PARTY_LICENSES.md', '.'), ('LGPL_COMPLIANCE.md', '.'), ('README.md', '.')]
binaries = []
hiddenimports = ['desktop_notifier.resources', 'desktop_notifier.backends.winrt', 'pydantic_core', 'aiosqlite']
datas += copy_metadata('genai_prices')
datas += copy_metadata('pydantic_ai_slim')
datas += copy_metadata('pydantic-ai-slim')
datas += copy_metadata('pydantic_ai')
datas += copy_metadata('pydantic-ai')
datas += copy_metadata('pydantic')
datas += copy_metadata('pydantic_graph')
datas += copy_metadata('griffe')
datas += copy_metadata('opentelemetry_api')
datas += copy_metadata('typing_inspection')
tmp_ret = collect_all('pydantic')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pydantic_core')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('boto3')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('botocore')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src\\subconscious\\desktop_ep.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['google.genai.types'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Subconscious',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:\\Users\\brian\\AppData\\Local\\Temp\\32251bdd-9533-4416-9a88-971af423187c',
    icon=['src\\subconscious\\assets\\favicon.ico'],
)
