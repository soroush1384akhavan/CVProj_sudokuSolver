# Windows Run Guide

## Backend

Open PowerShell:

```powershell
cd C:\path\to\sudoku-cv-windows-pytorch-opencv\backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If `torch` causes CUDA/DLL/page-file problems, install CPU-only PyTorch:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The project does not import torch on API startup. It only tries to load torch inside the digit recognition phase.

## Frontend

Open another PowerShell:

```powershell
cd C:\path\to\sudoku-cv-windows-pytorch-opencv\frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## If frontend install fails

Delete `node_modules` and `package-lock.json`, then reinstall:

```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json
npm install
npm run dev
```

This project uses Vite 5 to avoid the newer Rolldown native-binding issue on some Windows setups.


## Frontend dependency conflict fix

If `npm install` reports a Vite / @vitejs/plugin-react dependency conflict, delete `node_modules` and `package-lock.json`, then run `npm install` again. This project pins `vite` to `5.4.11` and `@vitejs/plugin-react` to `4.3.4` for Windows compatibility.

PowerShell:

```powershell
cd frontend
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item -Force package-lock.json -ErrorAction SilentlyContinue
npm install
npm run dev
```
