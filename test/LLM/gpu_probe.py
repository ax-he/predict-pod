#!/usr/bin/env python3
import json
import platform
import shutil
import subprocess
import sys

def run(cmd, shell=False):
    try:
        out = subprocess.check_output(cmd, shell=shell, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""

def try_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            out.append({
                "source": "PyTorch",
                "index": i,
                "name": name,
                "compute_capability": ".".join(map(str, cap))
            })
        return out
    except Exception:
        return []

def try_pynvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode() if hasattr(pynvml, "nvmlDeviceGetName") else str(pynvml.nvmlDeviceGetName(h))
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024**2)
            drv = pynvml.nvmlSystemGetDriverVersion().decode()
            out.append({
                "source": "NVML",
                "index": i,
                "name": name,
                "driver_version": drv,
                "memory_mb": mem
            })
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return []

def try_opencl():
    try:
        import pyopencl as cl
        out = []
        for p in cl.get_platforms():
            for d in p.get_devices():
                out.append({
                    "source": "OpenCL",
                    "platform": p.name.strip(),
                    "vendor": d.vendor.strip(),
                    "name": d.name.strip(),
                    "type": cl.device_type.to_string(d.type)
                })
        return out
    except Exception:
        return []

def linux_cmds():
    out = []
    # lspci 概览
    if shutil.which("lspci"):
        txt = run(["bash", "-lc", "lspci -nn | egrep 'VGA|3D|Display'"])
        if txt:
            out.append({"source":"lspci", "raw": txt})
    # nvidia-smi 细节
    if shutil.which("nvidia-smi"):
        q = "--query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader"
        txt = run(["bash","-lc", f"nvidia-smi {q}"])
        if txt:
            rows = []
            for line in txt.splitlines():
                parts = [p.strip() for p in line.split(",")]
                rows.append({
                    "name": parts[0],
                    "driver_version": parts[1] if len(parts)>1 else "",
                    "memory_mb": parts[2].replace(" MiB","") if len(parts)>2 else "",
                    "bus_id": parts[3] if len(parts)>3 else ""
                })
            out.append({"source":"nvidia-smi", "gpus": rows})
    return out

def windows_cmds():
    out = []
    # PowerShell CIM（推荐）
    ps = r"Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json"
    txt = run(["powershell","-NoProfile","-Command", ps])
    if txt:
        try:
            data = json.loads(txt)
            out.append({"source":"Win32_VideoController", "gpus": data})
        except Exception:
            out.append({"source":"Win32_VideoController", "raw": txt})
    # nvidia-smi
    if shutil.which("nvidia-smi"):
        q = "--query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader"
        txt = run(["nvidia-smi", q], shell=False)
        if txt:
            rows = []
            for line in txt.splitlines():
                parts = [p.strip() for p in line.split(",")]
                rows.append({
                    "name": parts[0],
                    "driver_version": parts[1] if len(parts)>1 else "",
                    "memory_mb": parts[2].replace(" MiB","") if len(parts)>2 else "",
                    "bus_id": parts[3] if len(parts)>3 else ""
                })
            out.append({"source":"nvidia-smi", "gpus": rows})
    return out

def mac_cmds():
    out = []
    # system_profiler
    if shutil.which("system_profiler"):
        txt = run(["/usr/sbin/system_profiler","SPDisplaysDataType","-json"])
        if not txt:
            # 某些系统在 /usr/sbin 之外
            txt = run(["system_profiler","SPDisplaysDataType","-json"])
        if txt:
            try:
                data = json.loads(txt)
                out.append({"source":"system_profiler", "displays": data})
            except Exception:
                out.append({"source":"system_profiler", "raw": txt})
    return out

def main():
    results = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": try_torch(),
        "nvml": try_pynvml(),
        "opencl": try_opencl(),
        "system": []
    }

    sysname = platform.system().lower()
    if "linux" in sysname:
        results["system"] = linux_cmds()
    elif "windows" in sysname:
        results["system"] = windows_cmds()
    elif "darwin" in sysname or "mac" in sysname:
        results["system"] = mac_cmds()

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
