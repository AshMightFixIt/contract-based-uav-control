# 🖥️ Setting Up on Your PC

## Step 1: Download All Files

**Option A: Download the entire folder**
1. Look for a "Download" button in Claude's interface
2. Download the `adaptive_drone_contracts` folder
3. Save it somewhere like `~/adaptive_drone_contracts` or `C:\adaptive_drone_contracts`

**Option B: If you can't download the folder, I can create a zip file**
- Let me know and I'll package everything into a single .zip file

---

## Step 2: Install Requirements

### On Linux/Mac:

```bash
# Navigate to project
cd ~/adaptive_drone_contracts

# Install Python dependencies
pip install numpy matplotlib --user

# OR if you have issues:
pip3 install numpy matplotlib --user
```

### On Windows:

```bash
# Navigate to project
cd C:\adaptive_drone_contracts

# Install Python dependencies
pip install numpy matplotlib

# OR
python -m pip install numpy matplotlib
```

**That's it!** No other dependencies needed. The system uses only:
- Python 3 (you probably already have this)
- NumPy (for math)
- Matplotlib (for plots)

---

## Step 3: Verify Setup

### On Linux/Mac:

```bash
cd ~/adaptive_drone_contracts
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
./test_all.sh
```

### On Windows (PowerShell):

```powershell
cd C:\adaptive_drone_contracts
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python test_all.sh  # Might need to run commands manually
```

### On Windows (Command Prompt):

```cmd
cd C:\adaptive_drone_contracts
set PYTHONPATH=%CD%\src;%PYTHONPATH%
python test_all.sh
```

---

## Step 4: Run the Demo

### On Linux/Mac:

```bash
cd ~/adaptive_drone_contracts
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python3 simulation_demo.py
```

### On Windows:

```powershell
cd C:\adaptive_drone_contracts
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python simulation_demo.py
```

**Expected output:**
- Console output showing flight simulation
- `simulation_results.png` file created
- `flight_log.json` and `flight_log_contracts.json` created

---

## Step 5: View Results

The simulation will create `simulation_results.png` in your project folder. Open it with any image viewer!

---

## 📁 Your PC Directory Structure

After download, you should have:

```
adaptive_drone_contracts/
├── START_HERE.md              ← Read this first!
├── README.md
├── simulation_demo.py         ← Main demo
├── test_all.sh
└── src/
    ├── contracts/
    │   └── contract_framework.py
    ├── estimation/
    │   └── contract_ekf.py
    ├── control/
    │   └── controllers.py
    └── adaptive_control_system.py
```

---

## 🐛 Troubleshooting

### "Module not found" errors:

**Make sure to set PYTHONPATH:**

Linux/Mac:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

Windows (PowerShell):
```powershell
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

### "No module named numpy/matplotlib":

```bash
pip install numpy matplotlib --user
```

### test_all.sh doesn't work on Windows:

Windows doesn't run .sh files easily. Instead, run each test manually:

```cmd
python src/contracts/contract_framework.py
python src/estimation/contract_ekf.py
python src/control/controllers.py
python src/adaptive_control_system.py
python simulation_demo.py
```

---

## ✅ Quick Verification

Run this to make sure everything works:

```bash
# Set Python path (adjust for your OS)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH  # Linux/Mac
# OR
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"  # Windows PowerShell

# Run demo
python3 simulation_demo.py  # or just 'python' on Windows

# Check for output files
ls -la simulation_results.png  # Should exist
ls -la flight_log.json         # Should exist
```

---

## 🎯 You're Ready!

Once you can run `simulation_demo.py` and it creates `simulation_results.png`, you're all set!

Then:
1. Read [START_HERE.md](START_HERE.md) for tomorrow's tasks
2. Start modifying code for Week 2

---

## 💡 Pro Tips

### Create a shell alias (Linux/Mac):

Add to your `~/.bashrc` or `~/.zshrc`:
```bash
alias drone='cd ~/adaptive_drone_contracts && export PYTHONPATH=$(pwd)/src:$PYTHONPATH'
```

Then just type `drone` to get ready!

### Create a batch file (Windows):

Create `setup.bat`:
```batch
@echo off
cd C:\adaptive_drone_contracts
set PYTHONPATH=%CD%\src;%PYTHONPATH%
```

Then run `setup.bat` whenever you want to work on the project.

---

## ❓ Still Having Issues?

If you run into problems:

1. **Check Python version**: `python --version` (need 3.7+)
2. **Check dependencies**: `pip list | grep -E "numpy|matplotlib"`
3. **Check file structure**: Make sure all files downloaded correctly
4. **Try running individual tests**: See which component fails

Let me know if you need help with any step!

---

**Last Updated**: November 24, 2025
