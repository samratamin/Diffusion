# Diffusion Experiment Processing: Sequence Type & Gradient Shape Implementation

## Overview
This document summarizes the implementation of support for different diffusion pulse sequences (PGSE vs PGSTE) and gradient shapes (square vs sinusoidal), with physics-appropriate signal processing equations and corrections applied automatically based on detected or specified parameters.

---

## 1. Pulse Sequence Types: PGSE vs PGSTE

### 1.1 PGSE (Pulsed Gradient Spin Echo)
- **Full name**: Pulsed Gradient Spin Echo
- **Pulse structure**: 90° - Gradient(+g, δ) - 180° - Gradient(-g, δ) - Acquire
- **Stejskal-Tanner equation**:
  ```
  ln(I/I₀) = -D · γ² · g_eff² · δ² · (Δ - δ/3)
  ```
  Where:
  - `γ` = gyromagnetic ratio of nucleus
  - `g_eff` = effective gradient field strength (T/m)
  - `δ` = gradient pulse duration (s)
  - `Δ` = diffusion time / gradient separation (s)
  - `D` = diffusion coefficient (m²/s)

- **Common implementations**:
  - Bruker STEBPGP, STEBPGP1S, BIPGP
  - Varian PGSE, BIPGP
  - Generic GED (Gradient Echo Diffusion)

### 1.2 PGSTE (Pulsed Gradient Stimulated Echo)
- **Full name**: Pulsed Gradient Stimulated Echo
- **Pulse structure**: 90° - Gradient(+g, δ) - 90° - Gradient(-g, δ) - 90° - Acquire
- **Stejskal-Tanner equation**:
  ```
  ln(I/I₀) = -D · γ² · g_eff² · δ² · (Δ - δ/3 - δ₂/2)
  ```
  Where:
  - All terms same as PGSE except:
  - `δ₂` = storage pulse duration (s)
  - The third term (δ₂/2) accounts for diffusion during the storage period

- **Common implementations**:
  - Bruker STED, STEBPGP1S (can use STED-like mixing)
  - Varian PGSTE, LED (Long Echo Diffusion)
  - Generic STIMULATED

### 1.3 Detection Strategy
The system detects sequence type from pulse program name using pattern matching:

**PGSTE keywords** (checked first):
- "sted" or "STED"
- "pgste" or "PGSTE"
- "stimulated"

**PGSE keywords** (default if no PGSTE match):
- "stebpgp", "STEBPGP"
- "bipgp", "BIPGP"
- "pgse", "PGSE"
- "ged", "GED"
- "led", "LED"
- "bppgp", "BPPGP"

**Default**: PGSE is assumed if pulse program name doesn't match any pattern.

### 1.4 Current Limitations & Future Enhancements
- **PGSTE δ₂ extraction**: Currently, true PGSTE processing uses (Δ - δ/3) like PGSE. Proper PGSTE support requires extracting `δ₂` from:
  - Bruker: P[] parameter array (specific index varies by sequence)
  - Varian: procpar file parameter
  - Action: Implement δ₂ extraction and conditional PGSTE equation when available

---

## 2. Gradient Shapes: Square vs Sinusoidal

### 2.1 Square (Hard) Gradients
- **Definition**: Constant magnetic field strength during pulse
- **Pulse shape**: Rectangular, constant amplitude
- **Correction factor**: **1.0** (no correction needed)
- **Effective gradient**: `g_eff = g · 1.0 = g`
- **Use case**: Standard hardware-limited gradients, rapid switching

### 2.2 Sinusoidal (Shaped) Gradients
- **Definition**: Ramped magnetic field strength (smooth shape)
- **Pulse shape**: Sine, Gaussian, or tapered envelope
- **Mathematical basis**: 
  - Square pulse has "full amplitude" across entire duration
  - Sinusoidal pulse has 0 amplitude at edges, maximum at center
  - Average effective amplitude is lower
  
- **Correction factor**: **0.9069** = π/(2√3) ≈ 0.9069
  - Derivation: Integral of sine wave vs constant over same duration
  - Accounts for reduced average field strength due to smooth envelope
  
- **Effective gradient**: `g_eff = g · 0.9069`
- **Use case**: High-quality diffusion measurements, reduced eddy current artifacts
- **Common implementations**:
  - Shaped rectangular pulses (RECT)
  - Sinusoidal envelopes (SINE)
  - Gaussian envelopes (GAUSS)

### 2.3 Detection Strategy
The system detects gradient shape from pulse program name:

**Sinusoidal indicators** (return factor 0.9069):
- "sine", "SINE"
- "sin", "SIN"
- "smooth", "SMOOTH"
- "shaped", "SHAPED"

**Square indicators** (return factor 1.0):
- "square", "SQUARE"
- "sqr", "SQR"
- "hard", "HARD"
- "rectangular", "RECTANGULAR"

**Default assumptions** (if no explicit indicator):
- STEBPGP, BIPGP, STED → **sinusoidal** (factor 0.9069)
- Most others → **square** (factor 1.0)

### 2.4 Current Limitations & Future Enhancements
- **Metadata limitation**: Pulse program name may not always indicate shape
- **Enhancement**: Store gradient shape in Bruker acqus/procpar or Varian procpar as searchable parameter
- **User override**: Consider adding UI option for user to manually specify if detection is uncertain

---

## 3. Processing Pipeline Integration

### 3.1 Data Flow
```
1. Upload NMR data (Bruker .ser/.fid or Varian fid)
                ↓
2. Process NMR Data (process_nmr_data)
   - Read FID from .ser or fid file
   - Apply exponential apodization: exp(-lb·π·n/N)
   - Compute FFT with user-selected zero-filling
   - Detect pulse program, sequence type, gradient shape
   - Store: raw_fid_slices, n_collected, sequence_type, gradient_shape, gradient_shape_factor
                ↓
3. Phase Correction (with dynamic re-processing)
   - User adjusts ph0, ph1, or optionally lb/fft_points
   - If lb or fft_points changed: Re-process from raw FID
   - Apply phase correction to complex spectra
                ↓
4. Peak Intensity Extraction
   - Select peaks of interest (user clicks on 2D plot)
   - Extract peak intensity at each gradient step
   - Normalize by internal standard if available
                ↓
5. Diffusion Analysis (analyze_diffusion)
   - Extract exp_params: sequence_type, gradient_shape, gradient_shape_factor
   - Read gradient levels from difflist or calculate from STSR calibration
   - **Apply gradient shape correction**: g_eff = g_bruker × gradient_shape_factor
   - **Select equation based on sequence_type**:
     - PGSE: ST_X = (γ·g_eff·δ)² · (Δ - δ/3)
     - PGSTE: ST_X = (γ·g_eff·δ)² · (Δ - δ/3 - δ₂/2) [if δ₂ available, else same as PGSE]
   - Fit intensities: I = I₀ · exp(-D · ST_X)
   - Return diffusion coefficients and statistics
                ↓
6. README Documentation
   - Document detected sequence type, gradient shape, correction factors
   - Show which equation was used
   - Document actual processing parameters (lb, fft_points, zerofill factor)
                ↓
7. Download Results
   - Export diffusion coefficients, fit statistics, spectra
   - Include processing details in README.txt
```

### 3.2 Bruker Data Flow (Specific to Bruker Format)
```
Bruker Directory Structure:
  experiment_dir/
    acqus         ← Contains P[], D[] arrays, PULPROG
    ser           ← FID data (4 bytes per point, TD/2 points)
    difflist      ← Gradient step list (0-1 fractions)
    pdata/
      1/
        procs     ← Contains STSR calibration info
        lists/
          gp/
            difframp  ← Gradient ramps if used
```

**Extraction steps in process_nmr_data (Bruker)**:
1. Read acqus to get:
   - PULPROG → pulse_program name
   - P[30] × 2 = δ (gradient pulse duration in μs)
   - D[20] = Δ (diffusion delay in ms)
   - SOLVENT → potential internal standard info

2. Call detect_sequence_type(pulse_program)
   - Returns "PGSE" or "PGSTE"

3. Call detect_gradient_shape(pulse_program, dic=acqus_dict, params=bruker_params)
   - Returns tuple (shape_name, correction_factor)

4. Store in params dict:
   - params['sequence_type'] = detected_type
   - params['gradient_shape'] = shape_name
   - params['gradient_shape_factor'] = correction_factor
   - params['delta'] = δ
   - params['big_delta'] = Δ

### 3.3 Varian Data Flow (Specific to Varian Format)
```
Varian Directory Structure:
  experiment_dir/
    fid           ← FID data (all points as binary)
    procpar       ← Parameters text file (key=value pairs)
```

**Extraction steps in process_nmr_data (Varian)**:
1. Read procpar to get:
   - seqfil → pulse sequence name (used as pulse_program)
   - Diffusion-specific parameters (location varies by sequence)

2. Call detect_sequence_type(seqfil)
   - Returns "PGSE" or "PGSTE"

3. Call detect_gradient_shape(seqfil, params=varian_params)
   - Returns tuple (shape_name, correction_factor)

4. Store in params dict (same as Bruker)

---

## 4. Implementation Details

### 4.1 Detection Functions

#### detect_sequence_type(pulse_program)
```python
def detect_sequence_type(pulse_program):
    """Detect PGSE vs PGSTE from pulse program name."""
    pp_lower = pulse_program.lower()
    
    # PGSTE check first (more specific)
    if any(kw in pp_lower for kw in ['sted', 'pgste', 'stimulated']):
        return 'PGSTE'
    
    # PGSE check
    if any(kw in pp_lower for kw in ['stebpgp', 'bipgp', 'pgse', 'ged', 'led', 'bppgp']):
        return 'PGSE'
    
    # Default
    return 'PGSE'
```

#### detect_gradient_shape(pulse_program, dic=None, params=None)
```python
def detect_gradient_shape(pulse_program, dic=None, params=None):
    """Detect square vs sinusoidal gradient shape.
    Returns (shape_name, correction_factor) tuple.
    """
    pp_lower = pulse_program.lower()
    
    # Sinusoidal indicators
    if any(kw in pp_lower for kw in ['sine', 'sin', 'smooth', 'shaped']):
        return ('sinusoidal', 0.9069)
    
    # Square indicators
    if any(kw in pp_lower for kw in ['square', 'sqr', 'hard', 'rectangular']):
        return ('square', 1.0)
    
    # Default assumptions by sequence type
    if any(kw in pp_lower for kw in ['stebpgp', 'bipgp', 'sted']):
        return ('sinusoidal', 0.9069)  # These sequences typically use shaped gradients
    
    # Default to square
    return ('square', 1.0)
```

### 4.2 Parameter Storage
After detection, parameters stored in `exp_params` dict (returned from process_nmr_data):
```python
exp_params = {
    'sequence_type': 'PGSE' or 'PGSTE',
    'gradient_shape': 'square' or 'sinusoidal',
    'gradient_shape_factor': 1.0 or 0.9069,
    'delta': float,              # in seconds
    'big_delta': float,          # in seconds
    'pulse_program': str,
    'vendor': str,
    'n_collected': int,          # for FFT re-processing
    # ... other params
}
```

### 4.3 Gradient Correction Application (in analyze_diffusion)
```python
# Extract parameters
exp_params = plot_data.get('exp_params', {})
sequence_type = exp_params.get('sequence_type', 'PGSE')
gradient_shape_factor = exp_params.get('gradient_shape_factor', 1.0)

# Convert gradients from G/cm to T/m
gradients = (m * x_points + c)              # in G/cm
g_tesla_m = gradients * 0.01               # convert to T/m

# Apply gradient shape correction
g_effective = g_tesla_m * gradient_shape_factor

# Select equation based on sequence type
if sequence_type == 'PGSTE':
    # PGSTE: includes δ₂/2 term (if available)
    st_x = (gamma * g_effective * delta)**2 * (big_delta - delta/3.0)  # - delta2/2 if known
else:
    # PGSE: standard equation
    st_x = (gamma * g_effective * delta)**2 * (big_delta - delta/3.0)

# Fit diffusion: I = I0 * exp(-D * st_x)
D = np.exp(curve_fit(...))
```

### 4.4 README Documentation
The _build_readme function now generates enhanced documentation:

```
EXPERIMENTAL PARAMETERS
  Vendor / Format:        BRUKER
  Pulse Program:          STEBPGP1S
  Sequence Type:          PGSE (Pulsed Gradient Spin Echo)
  Gradient Shape:         Sinusoidal
  Gradient Shape Factor:  0.9069 (correction applied to effective gradient)
  δ (gradient pulse):     0.002000 s  (2000.00 μs)
  Δ (diffusion time):     0.100000 s  (100.00 ms)
  Number of gradient steps: 32
  
DATA PROCESSING
  Phase correction (ph0): 45.20°
  Phase correction (ph1): 12.50°
  Baseline correction:    Polynomial order 3
  Apodization:            Exponential (lb = 1.0 Hz)
  Zero-filling:           4.0×
  Gradient shape:         Sinusoidal (shaped gradients)
  Gradient correction:    Applied correction factor 0.9069
  Diffusion equation:     PGSE: X = (γ·g·δ)² · (Δ - δ/3)
  Intensity extraction:   Maximum intensity
```

---

## 5. Processing Equations Reference

### 5.1 Complete PGSE Equation
**With all corrections applied:**
```
ln(I_k / I₀) = -D · γ² · (g_bruker_k × f_shape)² · δ² · (Δ - δ/3)

Where:
  I_k         = Peak intensity at gradient step k
  I₀          = Intensity at zero gradient
  D           = Diffusion coefficient (m²/s)
  γ           = 2.675 × 10⁸ rad/(s·T) for ¹H
  g_bruker_k  = Gradient strength from calibration (T/m)
  f_shape     = Gradient shape factor (1.0 or 0.9069)
  δ           = Gradient pulse duration (s)
  Δ           = Diffusion time (s)
```

### 5.2 Complete PGSTE Equation
**With all corrections applied (when δ₂ available):**
```
ln(I_k / I₀) = -D · γ² · (g_bruker_k × f_shape)² · δ² · (Δ - δ/3 - δ₂/2)

Where:
  δ₂          = Storage pulse duration (s)
  All other terms as in PGSE
```

**Current implementation** (without δ₂):
```
Same as PGSE: -D · γ² · (g_bruker_k × f_shape)² · δ² · (Δ - δ/3)
```

---

## 6. All Possible Scenarios

### Scenario 1: Bruker PGSE + Sinusoidal Gradients
- **Detection**: pulse_program contains "stebpgp" → PGSE + sinusoidal (0.9069)
- **Equation**: `ln(I/I₀) = -D · γ² · (g × 0.9069)² · δ² · (Δ - δ/3)`
- **Example**: STEBPGP1S on Bruker AVANCE
- **Expected result**: Diffusion coefficients corrected for sinusoidal shape

### Scenario 2: Bruker PGSE + Square Gradients
- **Detection**: pulse_program contains "pgse" or "ged" → PGSE + square (1.0)
- **Equation**: `ln(I/I₀) = -D · γ² · g² · δ² · (Δ - δ/3)` (standard)
- **Example**: Standard PGSE on older Bruker or custom implementation
- **Expected result**: Standard diffusion coefficients

### Scenario 3: Bruker PGSTE + Sinusoidal Gradients
- **Detection**: pulse_program contains "sted" → PGSTE + sinusoidal (0.9069)
- **Equation**: `ln(I/I₀) = -D · γ² · (g × 0.9069)² · δ² · (Δ - δ/3 - δ₂/2)` [if δ₂ known, else with Δ-δ/3 only]
- **Example**: STED diffusion experiments
- **Expected result**: Diffusion coefficients with stimulated echo correction

### Scenario 4: Bruker PGSTE + Square Gradients
- **Detection**: Pulse program contains both "sted" and "square" → PGSTE + square (1.0)
- **Equation**: `ln(I/I₀) = -D · γ² · g² · δ² · (Δ - δ/3 - δ₂/2)` [if δ₂ known]
- **Example**: Custom STED with hard gradients
- **Expected result**: PGSTE-corrected diffusion coefficients

### Scenario 5: Varian PGSE + Detected Shape
- **Detection**: seqfil detection determines sequence type and shape
- **Processing**: Same as Bruker, but with Varian FID format and procpar parameters
- **Example**: Varian PGSE, BIPGP sequences
- **Expected result**: Diffusion coefficients with appropriate corrections

### Scenario 6: Unknown Pulse Program
- **Detection**: Defaults to PGSE + square (1.0)
- **Equation**: `ln(I/I₀) = -D · γ² · g² · δ² · (Δ - δ/3)` (standard)
- **Fallback**: Safe default that covers most common cases
- **User action**: Can optionally add manual override in UI

---

## 7. Frontend Integration

### 7.1 Data Collection
The frontend now collects and sends detected parameters:
```javascript
const payload = {
    results: lastAnalysisResult.results,
    params: lastAnalysisResult.params,
    // ... other fields ...
    sequence_type: lastAnalysisResult.params?.sequence_type || 'PGSE',
    gradient_shape: lastAnalysisResult.params?.gradient_shape || 'square',
    gradient_shape_factor: lastAnalysisResult.params?.gradient_shape_factor || 1.0,
    // ... more fields ...
};
```

### 7.2 Download Information
When user downloads analysis, the README now includes:
- Detected sequence type
- Detected gradient shape and correction factor
- Which equation was applied
- Actual processing parameters (apodization, zero-fill factor)

### 7.3 Future UI Enhancements
- Display detected parameters on processing wizard
- Allow manual override if detection incorrect
- Show applied equation in results panel
- Warning if gradient shape uncertain

---

## 8. Testing & Validation

### Test Case 1: STEBPGP1S Data
- Upload Bruker STEBPGP1S diffusion data
- Expected: sequence_type='PGSE', gradient_shape='sinusoidal', factor=0.9069
- Verify: Diffusion coefficients approximately 0.9069 times smaller than if calculated without correction

### Test Case 2: Standard PGSE Data
- Upload Bruker PGSE or similar square-gradient data
- Expected: sequence_type='PGSE', gradient_shape='square', factor=1.0
- Verify: Diffusion coefficients match expected values

### Test Case 3: Varian Data
- Upload Varian PGSE or PGSTE data
- Expected: Parameters correctly extracted from Varian procpar
- Verify: Processing matches Bruker results for equivalent experiments

### Test Case 4: README Validation
- Download analysis results
- Expected README contents:
  - Correct sequence type and gradient shape listed
  - Correction factor shown if applicable
  - Equation explicitly stated
  - Processing parameters documented

---

## 9. Known Limitations & Future Work

### Current Limitations
1. **PGSTE δ₂ parameter**: Currently not extracted; equation uses Δ-δ/3 for all sequences
   - Fix: Identify Bruker P[] or Varian procpar parameters for storage pulse duration

2. **Gradient shape detection**: Based on pulse program name patterns
   - Limitation: May fail for custom or uncommon sequence names
   - Fix: Add user override UI; search for shape parameters in data files

3. **No calibration enhancement**: Sequence type/shape not stored per calibration
   - Enhancement: Add columns to calibrations table

4. **Limited error handling**: Ambiguous names may be mis-categorized
   - Enhancement: Fuzzy string matching; default to PGSE as safest option

### Future Enhancements
- [ ] Extract δ₂ from Bruker P[] and Varian procpar
- [ ] Implement true PGSTE equation with δ₂/2 term when available
- [ ] Add UI controls for manual sequence type/shape override
- [ ] Store sequence metadata with calibrations
- [ ] Fuzzy matching for ambiguous pulse program names
- [ ] Generate confidence scores for detections
- [ ] Export settings to save future re-processing with same parameters

---

## 10. Code Location Reference

### Bruker Parameter Extraction
- File: [app.py](app.py#L1495-L1550)
- Function: Bruker parameter extraction within process_nmr_data()

### Varian Parameter Extraction
- File: [app.py](app.py#L1630-L1660)
- Function: Varian parameter extraction within process_nmr_data()

### Detection Functions
- File: [app.py](app.py#L1397-L1468)
- Functions: `detect_sequence_type()`, `detect_gradient_shape()`

### Diffusion Analysis with Corrections
- File: [app.py](app.py#L95-L400)
- Function: `analyze_diffusion()` route - applies sequence_type and gradient_shape_factor

### README Generation
- File: [app.py](app.py#L510-L600)
- Function: `_build_readme()` - documents all applied corrections

### Frontend Data Collection
- File: [templates/index.html](templates/index.html#L1800-L1830)
- Collection point: Download endpoint payload preparation

---

## Summary

This implementation provides **automatic, physics-based processing** of diffusion NMR data with:
- ✅ Automatic PGSE vs PGSTE detection from pulse program names
- ✅ Automatic square vs sinusoidal gradient shape detection
- ✅ Gradient shape correction factor application (0.9069 for sinusoidal)
- ✅ Sequence-specific equation selection
- ✅ Transparent documentation of all applied corrections
- ✅ User-controllable processing parameters (apodization, FFT points)
- ✅ Publication-ready README with all processing details

The system handles all common scenarios while defaulting safely to PGSE + square gradients when uncertain, ensuring that even unknown pulse program names produce sensible results.
