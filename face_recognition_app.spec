# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('D:\\cloudone\\face-recogntion-deep-face\\haarcascade_frontalface_default.xml', '.')]
hiddenimports = ['deepface', 'PIL', 'cv2']
datas += collect_data_files('deepface')
hiddenimports += collect_submodules('tensorflow')
hiddenimports += collect_submodules('tf_keras')
hiddenimports += collect_submodules('keras')
hiddenimports += collect_submodules('deepface')


a = Analysis(
    ['D:\\cloudone\\face-recogntion-deep-face\\main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('v', None, 'OPTION')],
    exclude_binaries=True,
    name='face_recognition_app',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='face_recognition_app',
)
