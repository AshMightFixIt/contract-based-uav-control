# Contract-Based Adaptive UAV Control System - LaTeX Documentation

This folder contains comprehensive LaTeX documentation for the Contract-Based Adaptive UAV Control System.

## Compiling on Overleaf

1. **Create a new project** on Overleaf
2. **Upload all files** from this directory maintaining the folder structure:
   ```
   LaTeX_Database/
   ├── main.tex                 # Main document (compile this)
   ├── references.bib           # Bibliography
   ├── chapters/
   │   ├── titlepage.tex
   │   ├── nomenclature.tex
   │   ├── ch01_introduction.tex
   │   ├── ch02_architecture.tex
   │   ├── ch03_contracts.tex
   │   ├── ch04_estimation.tex
   │   ├── ch05_control.tex
   │   ├── ch06_planning.tex
   │   ├── ch07_safety.tex
   │   ├── ch08_integration.tex
   │   ├── ch09_results.tex
   │   ├── ch10_market.tex
   │   └── ch11_conclusion.tex
   ├── appendices/
   │   ├── appendix_code.tex
   │   ├── appendix_math.tex
   │   └── appendix_parameters.tex
   └── figures/
       └── (place any figures here)
   ```
3. **Set the main document** to `main.tex`
4. **Compiler settings**: Use pdfLaTeX
5. **Click Compile**

## Document Structure

### Main Chapters
- **Chapter 1**: Introduction - Motivation and system overview
- **Chapter 2**: System Architecture - Control pipeline, NED frame, data flow
- **Chapter 3**: Contract Framework - A/G contracts, Pacti library, composition
- **Chapter 4**: State Estimation - EKF with contract verification
- **Chapter 5**: Control Systems - PID and H-infinity controllers
- **Chapter 6**: Horizon Planning - Predictive planning with contracts
- **Chapter 7**: Safety Layer - Control Barrier Functions
- **Chapter 8**: System Integration - Complete adaptive control system
- **Chapter 9**: Experimental Results - Multi-waypoint mission demo
- **Chapter 10**: Market Analysis - Comparison with DJI, Skydio, PX4
- **Chapter 11**: Conclusion - Summary and future work

### Appendices
- **Appendix A**: Code Listings - Key implementation snippets
- **Appendix B**: Mathematical Derivations - Detailed proofs and equations
- **Appendix C**: System Parameters - Complete parameter reference

## Required Packages

All packages are standard and available on Overleaf:
- `tikz` with various libraries
- `pgfplots`
- `amsmath`, `amssymb`, `amsthm`
- `listings`, `algorithm`, `algpseudocode`
- `hyperref`, `cleveref`
- `natbib`
- `siunitx`
- `pifont`
- And others (see main.tex preamble)

## Custom Commands

The document defines several custom commands:
- `\importantbox{text}` - Purple highlighted box
- `\warningbox{text}` - Red warning box
- `\contractbox[title]{content}` - Contract specification box
- `\cmark`, `\xmark` - Checkmark and X symbols

## Notes

- The document uses the NED (North-East-Down) coordinate system where negative Z is altitude above ground
- All units are SI (meters, radians, seconds)
- The bibliography uses `natbib` with `plainnat` style

## Building Locally

If you prefer local compilation:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Contact

This documentation was generated for the Contract-Based Adaptive UAV Control System project.
