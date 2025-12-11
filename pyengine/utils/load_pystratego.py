import os
import glob
from importlib.machinery import ExtensionFileLoader
from pathlib import Path

_pystratego = None


def get_pystratego():
    global _pystratego
    if _pystratego is not None:
        return _pystratego

    # Find the project root (two levels up from current file)
    root = Path(__file__).resolve().parents[2]

    # Search for the compiled .so file
    so_candidates = list(root.rglob("pystratego*.so"))
    if not so_candidates:
        raise FileNotFoundError(f"No compiled pystratego .so found under {root}")

    # Use the first .so file found (you can add logic to prefer certain patterns)
    so_path = str(so_candidates[0])

    # Load it using ExtensionFileLoader
    _pystratego = ExtensionFileLoader("pystratego", so_path).load_module()
    return _pystratego


_old_pystratego = None


def get_old_pystratego():
    global _old_pystratego
    if _old_pystratego is not None:
        return _old_pystratego

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    old_pystratego_path = glob.glob(f"{root}/build/old_pystratego*.so")[0]
    _old_pystratego = ExtensionFileLoader("old_pystratego", old_pystratego_path).load_module()
    return _old_pystratego
