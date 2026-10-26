from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

TARGET = "tabular-lab-agent/DeepSeek"


class CredentialError(RuntimeError):
    pass


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(wintypes.BYTE)), ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR),
    ]


def _validate_key(key: str) -> str:
    key = key.strip()
    if not key or len(key) > 2560 or any(not 33 <= ord(char) <= 126 for char in key):
        raise CredentialError("API key must be a nonempty ASCII token without whitespace (max 2560 bytes).")
    return key


class WindowsCredentialStore:
    def __init__(self, target: str = TARGET):
        if os.name != "nt":
            raise CredentialError("Saved credentials require Windows; use DEEPSEEK_API_KEY on this platform.")
        self.target = target
        self.api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        self.api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.POINTER(ctypes.POINTER(_Credential))]
        self.api.CredReadW.restype = wintypes.BOOL
        self.api.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
        self.api.CredWriteW.restype = wintypes.BOOL
        self.api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.api.CredDeleteW.restype = wintypes.BOOL
        self.api.CredFree.argtypes = [ctypes.c_void_p]
        self.api.CredFree.restype = None

    def read(self) -> str | None:
        pointer = ctypes.POINTER(_Credential)()
        if not self.api.CredReadW(self.target, 1, 0, ctypes.byref(pointer)):
            error = ctypes.get_last_error()
            if error == 1168:
                return None
            raise CredentialError(f"Windows credential read failed (error {error}).")
        try:
            credential = pointer.contents
            if not 0 < credential.CredentialBlobSize <= 2560:
                raise CredentialError("Saved credential is invalid; run 'python -m tabular_lab auth login'.")
            return _validate_key(ctypes.string_at(credential.CredentialBlob,
                                                 credential.CredentialBlobSize).decode("utf-8"))
        except UnicodeDecodeError:
            raise CredentialError("Saved credential is invalid; run 'python -m tabular_lab auth login'.") from None
        finally:
            self.api.CredFree(pointer)

    def write(self, key: str):
        payload = _validate_key(key).encode("utf-8")
        buffer = ctypes.create_string_buffer(payload)
        credential = _Credential(Type=1, TargetName=self.target, UserName="DeepSeek API",
                                 CredentialBlobSize=len(payload),
                                 CredentialBlob=ctypes.cast(buffer, ctypes.POINTER(wintypes.BYTE)), Persist=2)
        try:
            if not self.api.CredWriteW(ctypes.byref(credential), 0):
                raise CredentialError(f"Windows credential save failed (error {ctypes.get_last_error()}).")
        finally:
            ctypes.memset(buffer, 0, len(buffer))

    def delete(self) -> bool:
        if self.api.CredDeleteW(self.target, 1, 0):
            return True
        error = ctypes.get_last_error()
        if error == 1168:
            return False
        raise CredentialError(f"Windows credential deletion failed (error {error}).")


def read_saved_key() -> str | None:
    return WindowsCredentialStore().read() if os.name == "nt" else None


def get_api_key() -> str | None:
    override = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    key = override or read_saved_key()
    return _validate_key(key) if key else None


def save_api_key(key: str):
    WindowsCredentialStore().write(key)


def delete_api_key() -> bool:
    return WindowsCredentialStore().delete()
