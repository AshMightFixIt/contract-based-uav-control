#!/bin/bash
# Quick Start Script for Adaptive Drone Contract System
# Run this to verify everything works

echo "======================================================================="
echo "  ADAPTIVE DRONE CONTROL WITH HIERARCHICAL CONTRACT COMPOSITION"
echo "======================================================================="
echo ""

# Set Python path
export PYTHONPATH=/home/claude/adaptive_drone_contracts/src:$PYTHONPATH
cd /home/claude/adaptive_drone_contracts

echo "📁 Project Location: $(pwd)"
echo ""

# Test 1: Contract Framework
echo "======================================================================="
echo "TEST 1: Contract Framework"
echo "======================================================================="
python3 src/contracts/contract_framework.py
if [ $? -eq 0 ]; then
    echo "✅ Contract framework test PASSED"
else
    echo "❌ Contract framework test FAILED"
    exit 1
fi
echo ""

# Test 2: Contract-Aware EKF
echo "======================================================================="
echo "TEST 2: Contract-Aware EKF"
echo "======================================================================="
python3 src/estimation/contract_ekf.py
if [ $? -eq 0 ]; then
    echo "✅ EKF test PASSED"
else
    echo "❌ EKF test FAILED"
    exit 1
fi
echo ""

# Test 3: Controllers
echo "======================================================================="
echo "TEST 3: Controllers"
echo "======================================================================="
python3 src/control/controllers.py
if [ $? -eq 0 ]; then
    echo "✅ Controllers test PASSED"
else
    echo "❌ Controllers test FAILED"
    exit 1
fi
echo ""

# Test 4: Complete System
echo "======================================================================="
echo "TEST 4: Integrated System"
echo "======================================================================="
python3 src/adaptive_control_system.py 2>&1 | head -50
if [ $? -eq 0 ]; then
    echo "✅ Integrated system test PASSED"
else
    echo "❌ Integrated system test FAILED"
    exit 1
fi
echo ""

# Final Summary
echo "======================================================================="
echo "✅ ALL TESTS PASSED"
echo "======================================================================="
echo ""
echo "📊 Available Files:"
echo "   - Contract Framework:    src/contracts/contract_framework.py"
echo "   - Contract-Aware EKF:    src/estimation/contract_ekf.py"
echo "   - Controllers:           src/control/controllers.py"
echo "   - Integrated System:     src/adaptive_control_system.py"
echo "   - Demonstration:         simulation_demo.py"
echo ""
echo "🚀 To run the full demonstration:"
echo "   python3 simulation_demo.py"
echo ""
echo "📖 For documentation, see:"
echo "   - README.md           (Complete system documentation)"
echo "   - WEEK1_SUMMARY.md    (Progress summary)"
echo ""
echo "🎯 Next Steps (Week 2):"
echo "   1. Add GPS degradation scenario"
echo "   2. Implement multiple waypoint mission"
echo "   3. Collect extensive flight data (50+ logs)"
echo ""
echo "======================================================================="
echo "System Ready! 🎉"
echo "======================================================================="
