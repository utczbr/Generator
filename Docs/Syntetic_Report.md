## 1. Pipeline Architecture & Domain Context

The generation of synthetic bar charts relies on a **code-guided synthesis framework**, a technique where underlying data tables and visual layout properties are programmatically linked to maximize logical consistency and statistical fidelity. This method directly reflects architectural paradigms utilized in modern chart reasoning and vision-language benchmarks, such as **ChartNet** (Kondic et al., 2026), **ChartMaster** (Liu et al., 2026), and **ChartGemma** (Masry et al., 2024).

The architecture separates the visualization workspace into two structural domains, ensuring that label taxonomies, statistical distributions, and chart aesthetics match real-world publications:

* **Scientific Domain:** Models realistic technical and physiological parameters based on biological assays, chemical measurements, and clinical variables. It utilizes domain-specific labels (e.g., *Molarity (mM)*, *Fold Change*, *Dose (mg/kg)*) and relies heavily on exact statistical variance overlays.
* **Business Domain:** Emulates corporate key performance indicators (KPIs), operational metrics, and economic records. It focuses on enterprise taxonomies (e.g., *Revenue (USD)*, *Customer Lifetime Value*, *Product SKU*) and structural trends like macro seasonality or market ceilings.

---

## 2. Statistical Data Models & Mathematical Formulas

Rather than using uniform random distributions, the data generation backbone samples from explicit mathematical profiles parameterized to conform to observed scientific and corporate phenomena.

### Scientific Data Models

#### Dose-Response (Hill Equation)

Models drug or stimulus effects across a range of log concentrations using literature-validated parameter bounds:


$$data = baseline + \frac{max\_response - baseline}{1 + 10^{(ec50 - log\_conc) \cdot hill\_slope}}$$

* **Heteroscedastic Error Modeling:** Measurement noise is non-uniform, scaling dynamically based on the current slope profile. The coefficient of variation ($CV$) peaks near the inflection points of the sigmoidal response curve:

$$cv = 0.05 + 0.10 \sqrt{response\_fraction \cdot (1 - response\_fraction)}$$


$$noise \sim \mathcal{N}(0, (data \cdot cv)^2)$$



#### Biological & Technical Replicates

Simulates variance across independent experimental groups. To match natural cell population deviations, a log-normal distribution is prioritized over a standard normal distribution:


$$\sigma = \sqrt{\ln(1 + CV^2)}$$

$$\mu = \ln(mean\_val) - 0.5\sigma^2$$

$$data \sim \exp(\mathcal{N}(\mu, \sigma^2))$$


*The target $CV$ varies dynamically based on the designated assay method (qPCR: 5–15%, Western Blot: 10–25%, Cell Assays: 15–35%).*

#### Exponential Decay

Represents pharmacokinetic clearance or radioisotope degradation parameters over time:


$$data = (initial\_value - baseline) \cdot e^{-k \cdot t} + baseline$$

$$k = \frac{\ln(2)}{half\_life}$$

* **Proportional Error Envelope:** Modeled using a combined error scheme incorporating both additive noise floor boundaries and proportional scale elements:

$$noise \sim \mathcal{N}(0, (data \cdot 0.08 + 0.02 \cdot max\_scale)^2)$$



#### Enzyme Kinetics (Michaelis-Menten)

Models initial reaction velocities of enzymatic processes under varying substrate levels:


$$data = \frac{V_{max} \cdot substrate}{K_m + substrate}$$


*Experimental noise is normally distributed relative to a stable assay variance threshold ($CV \in [0.05, 0.15]$).*

#### Spectroscopy / Chromatography Peak (Gaussian Profile)

Replicates sensor tracking from mass spectrometry or elution peaks:


$$data = amplitude \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}} + baseline$$

* **Poisson-like Photon Counting Noise:** Captures authentic instrument physics by evaluating noise as a function of current signal magnitude above baseline:

$$noise \sim \mathcal{N}\left(0, \left(0.3 \sqrt{|data - baseline|}\right)^2\right)$$



### Business Data Models

#### Multi-Component Seasonality & Trend

Simulates macroeconomic indicators or retail tracking across fiscal timelines:


$$data = base\_level + (trend\_slope \cdot x) + \sum_{m} \left[ amp_m \cdot \cos(freq_m \cdot x + phase_m) \right]$$


*The sub-component matrix $m$ evaluates concurrent annual periods ($\frac{2\pi}{12}$) and quarterly cycles ($\frac{2\pi}{4}$). Noise variance expands proportionally during peak seasonal waves.*

#### Pareto Distribution (80/20 Rule)

Simulates wealth distribution, inventory volume concentration, or regional performance disparities across discrete operational brackets:


$$data \sim \text{Pareto}(\alpha, x_m)$$

$$\alpha \in [1.05, 2.5]$$

#### Saturated Exponential Growth

Models subscriber acquisition curves, SaaS adoption cycles, or regional market entry up to carrying capacity limits:


$$raw = initial\_value \cdot e^{growth\_rate \cdot t}$$

$$data = raw \cdot \left(\frac{1}{1 + \frac{raw}{carrying\_capacity}}\right)$$

### Post-Processing Controls

To enforce mechanical plausibility, values are bounded using strict processing rules:

* **Overshoot Ceiling:** Extraneous values are truncated to a maximum of 105% of the calculated scale parameter, preventing disproportionate graph distortions.
* **Instrument Precision Limits:** Simulates digital rendering limits or human reading constraints by rounding data values to discrete decimal steps based on magnitude ($max \ge 1000 \rightarrow$ integer or 1 decimal place; $max < 10 \rightarrow 3$ decimal places).

---

## 3. Structural Layouts & Visual Styles

The rendering architecture dynamically handles several layout modalities depending on structural grouping needs (Li et al., 2024):

### Dimensional Configurations

* **Primary Value Axis:** Dictates graph alignment. Swaps dynamically between vertical bars (value scaling mapped on the Y-axis) and horizontal layouts (value scaling mapped on the X-axis) when dealing with dense categorical series (typically more than 6 groups).
* **Dual Y-Axis Scaling:** Evaluates distinct data sets on identical category axes. Independent left ($Y_1$) and right ($Y_2$) value limits handle multi-series comparisons (e.g., overlapping biological expression changes against direct tracking concentration units).

### Layout Strategies

* **Side-by-Side:** Arranges multiple data series sequentially across grouped categories using an indexed categorical displacement width ($\pm \frac{\text{width}}{2}$).
* **Stacked:** Layers data segments vertically on a shared bar footprint. The baseline coordinate for stacked segment $n$ resolves dynamically from the preceding cumulative total:

$$\text{bottom}_n = \sum_{i=1}^{n-1} \text{height}_i$$


* **Touching:** Maximizes bar width metrics ($width = 0.95$, $gap = 0.05$) to establish dense histograms or continuous value blocks.
* **3D Visual Anchors:** Applies structural drop-shadow geometries shifted by coordinates relative to figure rendering resolutions ($+2\text{dpi}, -2\text{dpi}$) to verify object contrast isolation.

### Hatch & Texture Variations

To represent distinct data components in monochoromatically constrained domains (e.g., grayscale journal prints), geometric texturing patterns can be mapped directly onto chart patches:

| Pattern ID | Code Token | Visual Representation |
| --- | --- | --- |
| **Hollow** | `none` | Clean border outline with zero-fill interior patch |
| **Dotted** | `..` | Dense stipple matrix layer |
| **Striped** | `//` | Right-leaning diagonal hatch stripes |
| **Crossed** | `xxxx` | Interlocking geometric mesh boundaries |

---

## 4. Contextual & Statistical Annotations

To support complex data extraction and spatial reasoning benchmarks, charts include advanced annotations mapping directly to calculated data parameters (Peng et al., 2024).

```
   Significance Bracket:  [      ***      ]
                          |               |
                    .-----------.   .-----------.
                    |           |   |     |     | <- Error Bar (Cap)
                    |           |   |     |     |
     Data Label ->  |   32.5    |   |  |--o--|  | <- Error Bar (Stem/Center)
                    |           |   |     |     |
                    |           |   |     |     |
                    |  Group A  |   |  Group B  |
              ------'-----------'---'-----------'------ Baseline (Y=0)

```

### Advanced Error Bars

Error configurations reflect custom experimental designs, varying based on the implied collection context:

* **Biological Replicates:** Computes standard deviation ($SD$) or standard error of the mean ($SEM$) over simulated experimental cohorts ($n \in [3, 8]$):

$$SD = value \cdot cv$$


$$SEM = \frac{SD}{\sqrt{n}}$$


* **Analytical Chemistry Assays:** Characterized by strict quality controls, forcing tight error boundaries ($CV \in [0.02, 0.08]$).
* **Demographic/Survey Data:** Displays structured margin of error parameters matching explicit target confidence intervals ($5\text{--}15\%$).

### Statistical Significance Overlays

Provides geometric markers verifying pair-wise group evaluations:

* **Bracket Modality:** Draws step-wise connecting vectors over targeted bar pairs. The bracket floor height clears the highest error margin inside the target zone, adding an elevation buffer ($10\text{--}20\%$ padding):

$$level = \max(\text{envelope}) \cdot (1 + \text{padding})$$



Labels use standard shorthand strings: `*`, ``, `***`, or `ns` (non-significant).
* **Letter Modality:** Assigns compact categorical alignment symbols (`a`, `b`, `c`, `d`) placed directly above the maximum bar or error coordinate.

### Data Labels & Positioning Matrix

To support robust character recognition workflows, scalar string values can be added to the chart. Labels track the coordinates of visual patches with precise positional adjustments:

* **Vertical Bars (Positive):** Located at $y = \text{top} + \text{offset}$, text alignment `va='bottom'`. If error bars are present, the label automatically moves to the edge of the highest error cap.
* **Vertical Bars (Negative):** Located at $y = \text{bottom} - \text{offset}$, text alignment `va='top'`.
* **Stacked Segments:** Positioned at the individual segment center or top boundary without cumulative summation interference, using precise segment coordinate tracking:

$$y_{\text{pos}} = \text{bottom}_n + \text{height}_n + \text{offset}$$



### Treatment Matrix Keys

Replaces general categorical labels in comparative drug or variant testing setups. It structures clean, black-and-white grids below the primary baseline, mapping binary condition criteria ($+$ or $-$ vectors) directly against the horizontal centers of the bars.

---

## 5. Themes, Aesthetics & Typography

A specialized thematic engine adjusts style parameters to mimic common software packages and publishing standards.

### Aesthetic Preset Configurations

| Theme ID | Base Color Space | Grid Style Configurations | Spine Borders |
| --- | --- | --- | --- |
| **Default** | Continuous perceptually uniform colormaps (`viridis`) | Light gray (`#CCCCCC`), dashed (`--`), thin | Left/Bottom enabled |
| **Excel** | Corporate muted accents (`#4472C4`, `#ED7D31`) | Solid white fill over gray background canvas | Left/Bottom enabled |
| **ggplot** | Saturated categorical hue blocks | Solid white lines over soft dark canvas (`#EBEBEB`) | All borders hidden |
| **Prism** | Bio-statistical qualitative palettes | Sparse dotted framework axis guides | Left/Bottom enabled |
| **Minimal** | Monochromatic black scale variables | Complete grid occlusion | Bottom baseline only |
| **Retro** | Vintage cream/earth tones (`#FFF8E7`, `#E4572E`) | Dense amber stipple grid markers | Left/Bottom enabled |

### Typography Options

Font rendering variables are dynamically sampled to ensure model generalization across multiple typefaces, weights, and scales:

* **Font Class Diversity:** Automatically samples distinct typefaces from Sans-Serif (e.g., Arial, DejaVu Sans, Liberation Sans), Serif (e.g., Times New Roman, Georgia), and Monospace families.
* **Proportional Text Scaling:** System titles use bold configurations scaled at 12–17pt; axis markers utilize 10–14pt; tick elements range between 8–12pt.
* **Dynamic Label Rotation:** Under dense category constraints, text blocks automatically rotate to 0°, 45°, or 90° angles to maintain structural legibility.

---

## 6. Recent Sources & Benchmarks

For details regarding current state-of-the-art methodology, refer to the following recent publications exploring code-guided visualization synthesis, tool-driven instruction tuning, and chart question-answering benchmarking datasets:

* **Kondic, J. (2026).** ChartNet: A million-scale, high-quality multimodal dataset for robust chart understanding. *arXiv preprint arXiv:2603.27064*.
* *Introduces massive code-guided synthesis pipelines to systematically bridge data frames, visualization primitives, and multi-step chart reasoning.*


* **Liu, C. (2026).** ChartMaster: Boosting MLLMs for chart analysis through data, perception, and reasoning optimization. *OpenReview / International Conference on Learning Representations*.
* *Deploys declarative asset rendering templates to convert complex records directly into executable plotting strings for model optimization.*


* **Meldrum, J., Suleiman, B., Rabhi, F., & Alibasa, M. J. (2025).** New money: A systematic review of synthetic data generation for finance. *arXiv preprint arXiv:2510.26076*.
* *Cited by: 6*
* *Reviews the implementation of hybrid top-down constraints alongside bottom-up statistical models to generate structured financial and business records.*


* **Li, Z., Jasani, B., Tang, P., & Ghadar, S. (2024).** Synthesize step-by-step: Tools, templates and LLMs as data generators for reasoning-based chart VQA. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 13613–13623. [https://doi.org/10.1109/cvpr52733.2024.01292](https://www.google.com/search?q=https://doi.org/10.1109/cvpr52733.2024.01292)
* *Cited by: 35*
* *Details template-based code abstractions and execution tracking frameworks to generate logically sound synthetic question-answering pairs.*


* **Masry, A., Thakkar, M., Bajaj, A., Kartha, A., Hoque, E., & Joty, S. (2024).** ChartGemma: Visual instruction-tuning for chart reasoning in the wild. *arXiv preprint arXiv:2407.04172*.
* *Cited by: 92*
* *Explores multi-modal tuning strategies combining underlying datatables from web repositories with programmatic charts for real-world chart extraction.*


* **Peng, Y.-H., Huq, F., Jiang, Y., Wu, J., Li, X. Y., Bigham, J. P., & Pavel, A. (2024).** DreamStruct: Understanding slides and user interfaces via synthetic data generation. *Lecture Notes in Computer Science*, 466–485. [https://doi.org/10.1007/978-3-031-72691-0_26](https://www.google.com/search?q=https://doi.org/10.1007/978-3-031-72691-0_26)
* *Cited by: 27*
* *Demonstrates the strength of initializing visual datasets using structural specifications converted into renderable code vectors.*

Programmatic generation of synthetic line charts relies on an automated pipeline that connects underlying statistical distributions with distinct layout standards. This pipeline is designed to enforce physical plausibility and logical consistency across all coordinate fields, providing clean datasets for modern computer vision and multi-modal benchmarks.

By analyzing the provided script architecture, the pipeline's structure—excluding post-rendering image effects—can be detailed across its core data engines, thematic settings, and keypoint tracking components.

---

## 1. Underlying Data Generation Models

The core dataset generation engine, contained in `generate_realistic_data` within `chart.py`, synthesizes line vectors using defined mathematical models that reflect actual phenomena observed in peer-reviewed scientific journals and business intelligence dashboards.

### Scientific Distributive Models

* **Dose-Response Curve:** Modeled using the sigmoidal Hill equation:



$$data = baseline + \frac{max\_response - baseline}{1 + 10^{(ec50 - log\_conc) \cdot hill\_slope}}$$



The log concentration vectors ($log\_conc$) range realistically from pM to mM ($-10$ to $-3$). Dynamic heteroscedastic noise is introduced because instrument error naturally scales near curve inflection points:



$$cv = 0.05 + 0.10 \sqrt{response\_fraction \cdot (1 - response\_fraction)}$$


$$noise \sim \mathcal{N}(0, (data \cdot cv)^2)$$


* **Biological & Technical Replicates:** Models log-normal variability across replicate groups (more realistic for cell populations than standard Gaussian distributions):



$$\sigma = \sqrt{\ln(1 + cv^2)}$$


$$\mu = \ln(mean\_val) - 0.5\sigma^2$$


$$data \sim \exp(\mathcal{N}(\mu, \sigma^2))$$



The coefficient of variation ($cv$) targets realistic experiment envelopes based on assay types (qPCR: $0.05\text{--}0.15$, Western blot: $0.10\text{--}0.25$, cell assays: $0.15\text{--}0.35$).


* **Exponential Decay:** Simulates chemical clearance or radioactive half-life:



$$data = (initial\_value - baseline) \cdot e^{-decay\_constant \cdot t} + baseline$$


$$decay\_constant = \frac{\ln(2)}{half\_life}$$



It applies a proportional error framework combining additive noise floors and proportional variables:



$$noise \sim \mathcal{N}(0, (data \cdot 0.08 + 0.02 \cdot max\_scale)^2)$$


* **Enzyme Kinetics:** Evaluates initial substrate conversion velocities via Michaelis-Menten equations:



$$data = \frac{vmax \cdot substrate}{km + substrate}$$


* **Gaussian Peak:** Replicates mass spectrometry or chromatography elution profiles:



$$data = amplitude \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}} + baseline$$



Poisson-like noise is injected to simulate authentic photon counting physics:



$$noise \sim \mathcal{N}\left(0, \left(0.3 \sqrt{|data - baseline|}\right)^2\right)$$



### Business Distributive Models

* **Seasonal Trends:** Evaluates macro timelines using joint periodic functions:



$$data = base\_level + (trend\_slope \cdot x) + \sum_{m} \left[ amp_m \cdot \cos(freq_m \cdot x + phase_m) \right]$$



Where components $m$ reflect annual seasonality ($freq = \frac{2\pi}{12}$) and quarterly cycles ($freq = \frac{2\pi}{4}$).


* **Saturated Exponential Growth:** Simulates user adoption curves using an explicit logistic ceiling:



$$raw = initial\_value \cdot e^{growth\_rate \cdot t}$$


$$data = raw \cdot \left(\frac{1}{1 + \frac{raw}{carrying\_capacity}}\right)$$


* **Pareto Distribution:** Tracks market share or wealth concentration:



$$data \sim \text{Pareto}(\text{shape}, x_m) \quad \text{where} \quad \text{shape} \in [1.05, 2.5]$$



---

## 2. Line Chart Visual Synthesis

The rendering workspace is executed via the `_generate_line_chart` function inside `chart.py`. This function configures the axes boundaries, handles the multi-series iteration loop, and processes geometric metadata before exporting artists to the compiler.

### Core Procedural Loop

1. **Series and Point Bounds:** The pipeline randomly instances the chart complexity, selecting between $1\text{--}4$ concurrent series lines and mapping discrete point lengths between $8\text{--}25$ points along the horizontal axis ($x = \text{np.arange}(\text{num\_points})$). Max scales are selected from discrete value steps: $[50, 100, 500, 1000]$.


2. **Visual Properties Matrix:** Each data trace is individually customized using randomly combined styling options:


* **Line Styles:** Chosen from `['-', '--', '-.', ':']`.


* **Markers:** Chosen from `[None, 'o', '^', 's', 'D', 'v', 'p', '*']`.


* **Weight Variables:** Linewidths vary uniformly between $1.5\text{--}3.0$ points.




3. **Data Pruning and Rendering:** Traces are evaluated through strict validation algorithms. Out-of-bounds metrics are clipped within a safe coordinate envelope ($-1.5 \cdot \max$ to $1.5 \cdot \max$), and NaN fields are converted to stable $0.0$ intercepts. Plots are rendered sequentially via `ax.plot()`.


4. **Label Context Assignment:** Text fields are randomly selected from specific label pools according to the domain setting:


* *Scientific pool items:* 'Wavelength (nm)', 'Concentration (μM)', 'Expression Level (a.u.)'.


* *Business pool items:* 'Sales ($M)', 'Churn Rate (%)', 'Monthly Active Users (MAU)'.




5. **Scaling Inversion Controls:** Minimum values across all traces are evaluated. If numbers fall strictly above $0$, axis transformations can upgrade from standard `linear` space to `log` or `symlog` scales with a default linear threshold (`linthresh=1.0`).



---

## 3. Themes, Grids, and Typography

The visual appearance of each chart is controlled by the preset matrices defined in `themes.py` and handled by helper modules within `chart.py`.

### Themes Preset Space

The layout parameters customize grid visibility, line thickness, and color palettes across various themes:

* **excel:** Uses a distinct grey canvas backdrop (`#F2F2F2`), thick solid white grid lines ($1.5$ pt), and strict corporate brand palette loops.


* **ggplot:** Replicates R-based rendering engines with a signature muted backdrop (`#EBEBEB`), white grid lines, and hides outer frame lines entirely.


* **prism:** Tailored for medical journals; enforces white backdrops, hides horizontal grid arrays completely, and sets directional ticks facing outwards.


* **minimal:** Strips away non-essential visual elements, using black scales, zero grid marks, and displaying only the bottom x-axis boundary.


* **retro / colorblind_friendly:** Maps customized color groups—such as Tol/Okabe-Ito compliance sets—over vintage canvases (`#FFF8E7`) to vary design constraints.



### Typography Variety

To prevent machine learning models from over-fitting to single typefaces, font properties vary dynamically:

* **Family Grouping:** Axes rotate between Sans-Serif options (Arial, DejaVu Sans, Liberation Sans) and Serif options (Times New Roman, Georgia, DejaVu Serif).


* **Proportional Sizing:** Chart titles adjust between $12\text{--}17$ pt with variable bold flags; axis text sets to $10\text{--}14$ pt; tick strings sit at $8\text{--}12$ pt.


* **Angle Rotation:** When category labels are closely packed, x-axis tick strings dynamically tilt at $0^\circ$, $45^\circ$, or $90^\circ$ angles to preserve readability.



---

## 4. Keypoint Detection & Structural Ground Truth

A major component of the pipeline is generating machine-readable annotations for ground-truth tracking, which is managed in `generator.py`. This structure supports both objective bounding-box tracking and deep coordinate extraction frameworks.

### Geometry and Structural Tracking

Traces are evaluated immediately after plotting to build data-space coordinate tables:

* **Inflection Marks:** Calculated by examining where the second derivative changes sign across three consecutive points:



$$d^2y = y_{i+1} - 2y_i + y_{i-1}$$



Points exceeding the threshold ($|d^2y| > 0.1 \cdot y_{range}$) are labeled as inflections.


* **Extrema Extraction:** Local peaks and valleys are extracted using `scipy.signal.find_peaks` over a smoothed curve filter to eliminate noise artifacts while preserving true geometric turning points.



### YOLO Pose Compliance (51-Keypoint Schema)

For object pose estimation tasks, line charts are mapped to a rigid 51-point skeleton format (`LINE_KEYPOINT_CONFIG`):

$$\text{Keypoint Indices:} \quad \underbrace{0}_{\text{Start}} \rightarrow \underbrace{1\text{--}25}_{\text{Resampled Path}} \rightarrow \underbrace{26\text{--}45}_{\text{Padded Inflections}} \rightarrow \underbrace{46\text{--}49}_{\text{Extrema Links}} \rightarrow \underbrace{50}_{\text{End}}$$

To enforce this strict format across varying data densities, the function `build_51_from_plotted` uses adaptive resampling:

```
                [ Adaptive Keypoint Normalization Matrix ]
                                   |
           -------------------------------------------------
          |                                                 |
   (Downsample Mode: N > 51)                       (Upsample Mode: N < 51)
          |                                                 |
Calculates Curvature Importance:                  Iterative Segment Splitting:
Evaluates turning angles and arc lengths.         Identifies the longest point span,
Discards low-scoring interior nodes,             splits it at the midpoint, and loops
preserving the 51 highest-priority coordinates.  until exactly 51 path points are met.

```

1. **Downsample Mode ($N > 51$):** Vertices are ordered from left to right. Curvature importance is evaluated by looking at the turn angles between adjacent segment vectors. The pipeline locks the endpoints and drops low-scoring interior nodes until exactly 51 coordinates remain.


2. **Upsample Mode ($N < 51$):** Points are evaluated in path order. The pipeline runs a heap queue to find the longest spatial segment, creates a new node at its midpoint, and dynamically recalculates the adjacent segments. This split loop repeats until the path contains exactly 51 sequential coordinates. This iterative approach distributes coordinates evenly across the entire trace, avoiding artificial point bundling.


3. **Coordinate Transformation:** The final 51-point path is passed to `extract_line_pose_annotations_fixed`, transformed from data space to absolute canvas pixels using the transformation matrix `ax.transData`, normalized to a $[0, 1]$ bounding envelope, and saved to the training directory.



---

## 5. Dataset Statistics and Recent Methods

When executed within `generator.py`, the pipeline aggregates these annotations into uniform outputs. Text objects—such as legend blocks or tick parameters—are checked against rendering boundaries via standard deviation calculations (`PIXEL_STD_DEV_THRESHOLD = 10`), removing hidden labels or empty elements before final output. Element boundaries and keypoint coordinates are exported as standardized YOLO formats and integrated into combined master index structures using `merge_json.py`.

This technical strategy of programmatic chart synthesis aligns with methodologies detailed in recent multi-modal and vision-language research:

* **ChartNet Dataset Framework (2026):** Uses programmatic data execution loops to link raw data frames with graphic primitives, building benchmarks for multi-step chart reasoning.
* **ChartMaster System Optimization (2026):** Deploys automated rendering templates to turn database outputs into executable visualization pipelines for vision-language models.
* **ChartGemma Benchmarking (2024):** Details multimodal instruction-tuning techniques that combine data tables with synthetic figures to match real-world chart layouts.
* **Synthesize Step-by-Step (Li et al., 2024):** Explores template-driven chart generation to create visually stable question-answering pairs without manual annotation errors.

Programmatic generation of synthetic box plot charts relies on an automated code-guided engine that establishes strict mappings between mathematical data frameworks and explicit graphical components. This methodology bypasses human annotation error by querying low-level graphic artists directly from execution threads, generating pristine ground-truth structures for multimodal vision-language models and scene-graph benchmarks.

---

## 1. Data Synthesis Models & Mathematical Formulations

Rather than populating charts with uniform random numbers, raw data traces are synthesized through continuous mathematical profiles that reflect physical constraints and heteroscedastic noise characteristics. Each box plot group represents a vector distribution containing $N$ discrete points (where $N \in [20, 50]$) bounded by a localized carrying capacity scale ($\text{max\_scale} = 100$).

The generation architecture pulls from several parameter-constrained equations to assemble the underlying data matrices:

### Sigmoidal Dose-Response Distribution (Hill Equation)

Simulates clinical response metrics across a sequence of logarithmic drug concentration benchmarks:


$$Y_i = \text{baseline} + \frac{\text{max\_response} - \text{baseline}}{1 + 10^{(\text{EC}_{50} - X_i) \cdot H}}$$

* **Parameters:** Concentration values $X_i \in [-10, -3]$ (pM to mM scale). The inflection median value $\text{EC}_{50} \sim \mathcal{U}(-8.5, -4.5)$. Hill cooperativity coefficient $H$ varies dynamically based on physiological target profiles: $H \sim \mathcal{U}(0.7, 1.3)$ for single baseline binding, $\mathcal{U}(1.3, 2.0)$ for cooperative transitions, or $\mathcal{U}(2.0, 3.5)$ for high cooperativity.
* **Heteroscedastic Error Modeling:** Measurement noise scales as a function of the slope magnitude, reaching maximum variance near the curve's inflection thresholds:

$$\text{CV}_i = 0.05 + 0.10 \sqrt{\left(\frac{Y_i - \text{baseline}}{\text{max\_response} - \text{baseline}}\right) \left(1 - \frac{Y_i - \text{baseline}}{\text{max\_response} - \text{baseline}}\right)}$$


$$\epsilon_i \sim \mathcal{N}(0, (Y_i \cdot \text{CV}_i)^2)$$



### Biological Technical Replicates (Log-Normal Distribution)

Replicates variance distributions typical of natural cellular expression levels, prioritizing an asymmetric log-normal skew over simple symmetric Gaussian patterns:


$$Y_i \sim \exp(\mathcal{N}(\mu, \sigma^2))$$

* **Parameters:** To bind sample bounds securely around a designated target mean ($\mu_{\text{target}}$) and target coefficient of variation ($\text{CV}_{\text{target}}$), parameters transform through the following mappings:

$$\sigma = \sqrt{\ln(1 + \text{CV}_{\text{target}}^2)}$$


$$\mu = \ln(\mu_{\text{target}}) - 0.5\sigma^2$$


* The targeted $\text{CV}$ matches literature envelopes according to specific testing modalities (qPCR assays: $5\text{--}15\%$, Western Blots: $10\text{--}25\%$, dynamic cell microplates: $15\text{--}35\%$).

### Pharmacokinetic Clearance (Exponential Decay)

Simulates half-life elimination tracking curves:


$$Y_i = (Y_0 - \text{baseline}) \cdot e^{-k \cdot t_i} + \text{baseline}$$

* **Parameters:** Decay constants scale from explicit target half-lives ($t_{1/2} \in [0.5, 72]$ hours): $k = \frac{\ln(2)}{t_{1/2}}$. A proportional noise envelope overlays the sequence to scale variance with current concentrations: $\epsilon_i \sim \mathcal{N}(0, (Y_i \cdot 0.08 + 0.02 \cdot \text{max\_scale})^2)$.

---

## 2. Layout Formats & Visual Mapping

The core layout generator handles the rendering via the `_generate_boxplot_chart` function. The spatial properties of the workspace adjust dynamically between two principal layouts:

```
[ Vertical Layout Orientation ]           [ Horizontal Layout Orientation ]

             Axis Title (Y)                            Group Category (Y)
                 ^                                             ^
                 |                                             |
             |---|---|  <- Cap                         |--[  |  ]--|
                 |                                             |
                 |     <- Whisker                              |
             .-------.                                      .-------.
             |       |  <- Q3 (75th)                        |   |   |
             |-------|  <- Median Line                      |   |   |
             |       |  <- Q1 (25th)                        |   |   |
             '-------'                                      '-------'
                 |                                      Q1  Median  Q3
                 |                                    
                 o      <- Outlier (Flier)            
                 v                                             v
                 ------------------->                          ------------------->
                  Group Category (X)                            Axis Title (X)

```

### Dimensional Layout Adjustments

* **Vertical Orientation:** Categories align horizontally across discrete index points. The descriptive numerical metrics map directly onto the vertical Y-axis. This is the standard operational path ($85\%$ activation chance) and handles between $3\text{--}8$ concurrent data columns.
* **Horizontal Orientation:** Implements a alternative layout matrix ($15\%$ activation chance). Categories stack sequentially down the vertical plane, forcing numerical metric evaluation along the horizontal X-axis. This format handles higher group allocations, scaling smoothly between $6\text{--}12$ boxes.

### Internal Architectural Implementation

The layout is instantiated programmatically through Matplotlib's boxplot API, which splits the structural visual patches into an explicit dictionary hierarchy:

```python
bp = ax.boxplot(datas, patch_artist=True, widths=box_width, vert=not is_horizontal, flierprops=...)

```

---

## 3. Structural Components & Bounding Annotation Specs

During code execution, the pipeline extracts pixel-perfect bounding coordinates for every constituent piece of the graphic canvas. These regions correspond directly to a rigid multi-class schema (`CLASS_MAP_BOX`) configured for object detection and visual parser pipelines:

* **`box` (Class 1):** The rectangular perimeter mapping the Interquartile Range ($\text{IQR} = Q_3 - Q_1$). The top and bottom edges are clamped exactly to the 75th percentile ($Q_3$) and 25th percentile ($Q_1$), respectively.
* **`range_indicator` (Class 4):** A single unified box built by taking the geometric union of the top and bottom whiskers and their relative terminal caps. Whiskers project out to the furthest data points falling within $1.5 \times \text{IQR}$ from the box margins.
* **`median_line` (Class 7):** The interior median segment tracking the 50th percentile ($Q_2$). Because single line primitives return near-zero height or width attributes in raw coordinate buffers, the pipeline applies an active structural padding mechanism ($3$ display pixels) to establish a clear region format.
* **`outlier` / `flier` (Class 9):** Individual sample points that fall outside the $1.5 \times \text{IQR}$ boundary limits. Fliers are drawn using customized geometric markers (circles `'o'`, stars `'*'`, or diamonds `'D'`) at precise coordinate transforms.

---

## 4. Contextual & Statistical Annotations

Advanced statistical overlays can be added to the chart to enhance layout complexity and evaluate model extraction capabilities under dense visual environments:

### Individual Sample Jitter Overlays

To preserve absolute density visibility within monochromatic profiles, a scatter overlay can be activated ($20\%$ probability in scientific environments). Individual points from the raw data arrays are plotted directly over the respective boxes. To prevent cluster stacking along a single straight path, points are dispersed across the category column center using a tight normal distribution constraint:


$$X_{\text{jitter}} \sim \mathcal{N}\left(\text{Group\_Center},\, 0.04^2\right)$$

### Statistical Significance Overlays

To denote pair-wise mathematical testing conclusions, significance annotations are drawn over specified box pairs ($50\%$ activation probability):

* **Bracket Notation:** Connects target box centers using structured step-wise lines. The horizontal beam clearance level is computed directly from the maximum whisker cap height found within the span index, adding a clear padding buffer to avoid overlap:

$$\text{Level} = \max(\text{whiskers}_{\text{start}\dots\text{end}}) \times (1 + \mathcal{U}(0.10,\, 0.20))$$



The horizontal segment is centered with key indicator strings (`*`, ``, `***`, or `ns`).
* **Letter Taxonomy:** Compact text identifiers (`a`, `b`, `c`, `d`) are positioned at a fixed vertical offset ($0.05 \times Y_{\text{max}}$) above the maximum whisker cap of each group.

---

## 5. Themes, Aesthetics & Typography

The visual style of the box plot adjusts dynamically using structural design rules defined in the thematic architecture:

### Thematic Presets Matrix

| Theme ID | Base Color Space | Patch Style | Grid & Axis Boundaries |
| --- | --- | --- | --- |
| **`scientific`** (`prism` / `default`) | Monochromatic / Grayscale or high-contrast qualitative scales | **Hollow Modality:** Solid line perimeters, transparent fill interiors (`facecolor='none'`). | Axis grids are hidden or restricted to light dashed horizontal indicators. |
| **`corporate`** (`excel` / `powerpoint`) | Muted corporate accents (`#4472C4`, `#ED7D31`) | **Filled Modality:** Saturated fill faces bounded by clean black borders. | Soft solid grey or white horizontal backing panels. |
| **`retro`** | Vintage warm palettes (`#E4572E`, `#17BEBB`) | Filled face color blocks. | Soft cream backing grid panels (`#FFF8E7`). |

### Typography & Label Variety

* **Typeface Sampling:** Typefaces rotate systematically across multiple font families to maximize model generalization:
* *Sans-Serif:* Arial, DejaVu Sans, Liberation Sans.
* *Serif:* Times New Roman, Georgia, DejaVu Serif.


* **Variable Text Scales:** Layout components scale dynamically across designated font ranges: chart titles use $12\text{--}17\text{pt}$ weights, axis titles scale between $10\text{--}14\text{pt}$, and individual tick values are set to $8\text{--}12\text{pt}$ options.
* **Dynamic Label Rotation:** When category labels are dense, x-axis labels automatically rotate ($0^\circ$, $45^\circ$, or $90^\circ$) to keep text clear and prevent overlapping.

---

## 6. Recent Sources & Benchmarks

Recent vision-language research demonstrates that directly extracting data metrics from abstract graphical figures (such as box plots without text overlays) remains difficult for general vision-language architectures. To address this challenge, current benchmarks leverage programmatic, code-guided synthesis loops to fine-tune and evaluate models on dense chart reasoning tasks:

* **Borisova, L. et al. (SciVQA 2025).** *SciVQA 2025: Overview of the First Scientific Visual Question Answering Shared Task*. ACL Anthology.
* Establishes comprehensive evaluations targeting multi-modal scientific documents, highlighting structural chart types like box plots where data tracking across axis grids is required.


* **Masry, A. et al. (ChartQAPro 2025).** *ChartQAPro: A New Benchmark for Complex Chart Question Answering*. Findings of ACL.
* Introduces unannotated and complex statistical visualization benchmarks to evaluate model perception against varied structural configurations, listing explicit error patterns in spatial coordinate alignment.


* **Wang, X. et al. (2025).** *Chain of Functions: A Programmatic Pipeline for Fine-Grained Chart Reasoning Data Generation*. arXiv / ACL Anthology.
* Details the implementation of program-based functional discovery and code execution trees as an intermediate layer to create structured chart-to-table datasets.


* **Pramanick, A. et al. (InterChart 2026).** *INTERCHART: Benchmarking Visual Reasoning Across Decomposed and Distributed Chart Information*. IJCNLP-AACL.
* Deploys deterministic query architectures and programmatic chart-rendering scripts to create multi-tier synthetic benchmarks, tracking localized visual elements and cross-visual dependencies.

The generation of synthetic heatmap charts without post-rendering image effects relies on a code-guided synthesis architecture that programmatically orchestrates data-generation matrix models, degradation filters, graph seriation operations, and layout compilation frameworks. By directly linking low-level graphical artists with statistical properties, this technique enables the production of pixel-perfect annotation tables required to train and evaluate multi-modal vision models and visual document parsers.

---

## 1. Pipeline Architecture & Domain Context

The heatmap synthesis pipeline orchestrates data generation via a unified configuration dictionary (`HEATMAP_GENERATION_CONFIG`), which defines strict probabilistic weights for structural matrix archetypes, normalization mappings, missing data injectors, and cell seriation systems.

To simulate authentic data environments, the pipeline separates generation variables across specialized contextual profiles:

* **Scientific Profile (`genomic_expression_heatmap`, `pharmacokinetics_heatmap`):** Emulates high-throughput biological screens, such as next-generation sequencing (NGS) transcript read panels or mass spectrometry profiles. Rows and columns map domain-specific label taxonomies (e.g., *Gene Symbol*, *Pathway*, *Wavelength (nm)*, *Time Post-dose (h)*) and implement row-wise standardization layouts.
* **Business Profile (`cohort_retention_heatmap`):** Replicates operational metrics and corporate dashboards. Labels map corporate data fields (e.g., *Product SKU*, *Sales Region*, *Customer Lifetime Value*, *Fiscal Quarter*) and enforce clear numeric formatting wrappers (e.g., percentages or monetary scales).

---

## 2. Matrix Structural Paradigms & Mathematical Formulations

Rather than initializing heatmaps with uniform random numbers, the grid matrices ($M \in \mathbb{R}^{R \times C}$, where rows $R \in [8, 30]$ and columns $C \in [8, 30]$) are synthesized through continuous mathematical algorithms:

### Davies-Higham Positive Semi-Definite (PSD) Correlation Matrices

Generates authentic correlation matrices from parameterized eigenvalue spectrum profiles:

* **Eigenvalue Selection:** A random vector is drawn from a Dirichlet distribution under varying concentration constraints:

$$\lambda \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_D) \times D$$



*Where $\alpha = 0.1$ simulates tight, high-strength correlation clusters, $\alpha = 10.0$ yields diffuse, low-strength interactions, and $\alpha = 1.0$ constructs a uniform eigenvalue spectrum*.
* **Correlation Generation:** The generated eigenvalue array is scaled to sum to the dimension $D$, and a random correlation matrix is evaluated via orthogonal transformations:

$$M = \text{random\_correlation.rvs}(\lambda)$$



### LKJ Vine Correlation Models

Samples correlation matrices uniformly over the space of positive definite matrices using partial correlation trees governed by an explicit shape parameter $\eta \in [0.6, 3.0]$:


$$\beta_{\text{param}} = \eta - 1.0 + \frac{D}{2}$$


Partial correlations are sampled iteratively from a transformed Beta distribution, $P_{k,i} = (2 \times \text{Beta}(\beta_{\text{param}}, \beta_{\text{param}})) - 1.0$, and mapped recursively to reconstruct the off-diagonal elements of the symmetric matrix.

### Coherent Additive & Multiplicative Biclusters

Simulates synchronized features embedded inside background noise fields:

* **Additive Coherent Block:** Replicates standard checkerboard expression clusters using row and column effects over a base value $\mu$:

$$M_{i,j} = \mu + \alpha_i + \beta_j + \mathcal{N}(0, \sigma^2)$$



*Where $\alpha_i \in [2.0, 5.0]$ represents the row effect vector and $\beta_j \in [2.0, 5.0]$ represents the column effect vector*.
* **Multiplicative Coherent Block:** Models synergetic or exponential cross-talk within sub-matrices:

$$M_{i,j} = \mu \times \alpha_i \times \beta_j + \mathcal{N}(0, \sigma^2)$$



*Where $\alpha_i \in [1.1, 2.0]$ and $\beta_j \in [1.1, 2.0]$ act as scale multipliers*.

### Spatial Coherence (Perlin & Fractal Noise Matrices)

Models smooth spatial gradients (such as geographic distributions or sensor fields). A grid mesh is evaluated against randomized directional gradients, smoothed using a quintic fade function:


$$f(t) = 6t^5 - 15t^4 + 10t^3$$


For complex multi-scale terrain surfaces, Fractal noise accumulates successive octaves modulated by a persistence value $p \in [0.35, 0.65]$:


$$M_{\text{fractal}}(x, y) = \sum_{k=0}^{\text{octaves}-1} p^k \cdot M_{\text{Perlin}}(2^k \cdot x, \, 2^k \cdot y)$$

### Spatio-Temporal Matrices (SARIMA with Spatial Autoregression)

Generates temporal series lines across matrix rows with structural columns linked via a Spatial Autoregressive (SAR) weight matrix:

1. **Temporal Core:** Matrix steps are generated along the temporal plane using seasonal auto-regressive moving average components:

$$B_{t, \cdot} = \text{SARIMAX}(\text{order}, \, \text{seasonal\_order})$$


2. **Spatial Transformation:** Spatial cross-talk is injected across series elements using a distance-based weight matrix $W$, where $W_{i,j} = \frac{1}{|i-j|}$ for $i \neq j$ (row-normalized so that $\sum_j W_{i,j} = 1$):

$$M = B \cdot (I - \rho W)^{-T}$$



*Where $\rho \in [0.15, 0.60]$ specifies the spatial dependence strength multiplier*.

### Block-Diagonal Communities & Toeplitz Decay Layouts

* **Block-Diagonal Communities:** Combines isolated dense sub-matrices along a global diagonal workspace using `scipy.linalg.block_diag`, with adjustable cell densities to emulate isolated group structures.
* **Toeplitz Autoregressive Matrix:** Populates structured matrix fields according to an exponential distance decay model governed by parameter $\rho \in [0.15, 0.60]$:

$$M_{i,j} = \rho^{|i - j|} + \mathcal{N}(0, \sigma^2)$$



---

## 3. Data Degradation & Variance Simulation

Before compilation, raw value matrices pass through specialized filtering modules to simulate data imperfections and physical instrumentation noise bounds:

### Missing Data Patterns (MCAR & MNAR Filters)

* **Missing Completely at Random (MCAR):** Values are dropped uniformly across the matrix layout based on a defined rate variable $\omega \in [0.02, 0.10]$.
* **Missing Not at Random (MNAR) Logistic Projection:** Missingness tracks a sigmoidal probability dependent on cell intensity:

$$P(\text{Missing}_{i,j}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot M_{i,j})}}$$


* **MNAR Quantile Censorship:** Truncates all values falling beyond a targeted data quantile ($q \in [0.80, 0.95]$) to simulate top-end ceiling clipping.
* **MNAR Detection Limits (LLOD / ULOQ):** Simulates instrument bounds by discarding entries outside a Lower Limit of Detection ($LLOD$) or Upper Limit of Quantification ($ULOQ$) threshold.

### Advanced Noise Frameworks

* **Proportional Heteroscedastic Noise:** Scales variance dynamically with data magnitude:

$$\epsilon_{i,j} \sim \mathcal{N}\left(0, \, (\gamma |M_{i,j}| + \delta)^2\right) \quad \text{where} \quad \gamma \in [0.05, 0.15]$$


* **Exponential Heteroscedastic Noise:** Models exponential error expansion under intense sample loads:

$$\epsilon_{i,j} \sim \mathcal{N}\left(0, \, (\delta \cdot e^{\alpha |M_{i,j}|})^2\right) \quad \text{where} \quad \alpha \in [0.02, 0.08]$$


* **Poisson-Like Noise:** Replicates count-based shot noise variance common in sensory instruments:

$$\epsilon_{i,j} \sim \mathcal{N}\left(0, \, k \cdot |M_{i,j} + \text{shift}|\right) \quad \text{where} \quad k \in [0.4, 2.0]$$


* **Parametric Variance with Pareto Outliers:** Variance follows a stable polynomial power function ($\sigma^2 = \alpha + \beta |M|^\gamma$). Extreme data spikes are injected at sparse coordinates using skewed Pareto tail distributions scaled by a factor of $6\text{--}14\times$.

---

## 4. Ordering & Structural Seriation Systems

To benchmark chart-parsing networks against complex layouts, heatmaps can be sorted using graph seriation algorithms to discover hidden visual matrices:

### Optimal Leaf Ordering (OLO)

Evaluates hierarchical clustering trees to sort adjacent rows and columns, maximizing structural proximity along matrix vectors. The pipeline builds a pairwise distance matrix via a specified metric (e.g., Euclidean distance), evaluates clusters using Ward's minimum variance linkage method, and flips internal branch nodes to find an optimal leaf progression layout:


$$\arg\min_{\pi} \sum_{i=1}^{N-1} d(X_{\pi(i)}, \, X_{\pi(i+1)})$$

### Spectral Fiedler Vector Ordering

Sorts matrix arrays using a graph laplacian layout approach:

1. **Similarity Construction:** Converts row arrays into an RBF similarity graph kernel:

$$A_{i,j} = e^{-\gamma \|X_i - X_j\|^2}$$


2. **Laplacian Derivation:** Computes the unnormalized graph Laplacian operator from a diagonal degree matrix $D$ ($D_{i,i} = \sum_j A_{i,j}$):

$$L = D - A$$


3. **Eigenvalue Evaluation:** Solves the symmetric eigensystem $L\mathbf{v} = \lambda D\mathbf{v}$. The eigenvector corresponding to the second smallest eigenvalue (the *Fiedler Vector*) is isolated, and its sorting indexes dictate the matrix seriation index sequence.

### Normalization Operations

Data ranges are standardized using specialized normalization methods before rendering:

* **Row-Wise Z-Score:** Standardizes profiles across rows: $M'_{i,j} = \frac{M_{i,j} - \mu_i}{\sigma_i + \epsilon}$
* **Global MinMax:** Squashes values into a consistent unit interval: $M'_{i,j} = \frac{M_{i,j} - \min(M)}{\max(M) - \min(M)}$
* **Robust IQR Scaler:** Protects data ranges from outlier compression: $M'_{i,j} = \frac{M_{i,j} - \text{median}_j}{\text{IQR}_j + \epsilon}$
* **Log10 Scale Transformation:** Converts exponential values into linear space: $M'_{i,j} = \text{sign}(M_{i,j}) \times \log_{10}(|M_{i,j}| + \epsilon)$

---

## 5. Layout Rendering & Geometric Ground-Truth Mapping

The heatmap is programmatically rendered using Matplotlib's `pcolormesh` API, which creates a highly detectable `QuadMesh` graphical artist component on the canvas:

```
                     [ Grid Transformation Canvas ]
                                   |
         -----------------------------------------------------
        |                                                     |
 [ Primary Plot Axis ]                              [ Canvas Display Buffer ]
  ax.pcolormesh(X, Y, M)                             fig.canvas.draw()
  Creates a QuadMesh primitive                        Generates pixel coordinates.
        |                                                     |
         -----------------------------------------------------
                                   |
                     [ Annotation Parsing Engine ]
                       get_granular_annotations()
                                   |
               Loops over row and column index junctions.
                                   |
               - corners = coords[i,j], coords[i+1, j+1]...
               - px_bbox = ax.transData.transform(corners)
                                   |
         -----------------------------------------------------
        |                                                     |
 [ Normalized YOLO Specs ]                        [ Comprehensive JSON Tree ]
  Outputs central (x, y, w, h)                     Maps 'cell', 'colorbar', and text
  bounding logs for cell patches.                  labels with pixel-level precision.

```

### Bounding Box Resolution Tracking

The pipeline parses these graphical objects within `get_granular_annotations` via the following steps:

1. **Artist Validation:** The script queries the scene graph, isolates elements matching `isinstance(artist, QuadMesh)`, and extracts cell vertex information using `artist.get_coordinates()`.
2. **Corner Sorting Loop:** For every row $i$ and column $j$ in the active matrix grid, the script extracts the four bounding boundary vertices:

$$\text{Corners} = \left[ \mathbf{c}_{i,j},\, \mathbf{c}_{i+1,j},\, \mathbf{c}_{i+1,j+1},\, \mathbf{c}_{i,j+1} \right]$$


3. **Coordinate Transformation:** Vertices are mapped from data space to absolute image display pixels using the axis transformation matrix `ax.transData.transform(cell_corners)`. Min-max coordinate limits define the cell's bounding boundaries:

$$x_0, y_0 = \min(\text{Display}), \quad x_1, y_1 = \max(\text{Display})$$



These boundaries are normalized to a $[0, 1]$ coordinate scale to export structured labels (`CLASS_MAP_HEATMAP`).

### Auxiliary Elements Extraction

* **Color Bar Extraction (`color_bar`, Class 3):** Isolated by scanning all figure axes objects. Axes that do not match the primary plot axis are filtered using aspect ratio thresholds ($\frac{\text{width}}{\text{height}} < 0.3$ for vertical arrays or $> 3.0$ for horizontal bars) to yield exact color bar bounding parameters.
* **Color Bar Structural Labels:** Explicit text annotations are linked to the color bar to capture its title string (`'color_bar_title'`) and numeric scalar markings (`'color_bar_label'`).
* **Numeric Cell Value Labels (`data_label`, Class 6):** Text items are drawn at cell center offsets ($c+0.5, \, r+0.5$) using dynamically selected cell format rules (e.g., `"{:.2f}"`, `"{:.0%}"`, or scientific templates). Font colors are adjusted to black or white based on cell brightness values to ensure clear text legibility.

---

## 6. Themes, Aesthetics & Typography Variations

Visual attributes are adjusted using configured styling profiles defined in `themes.py`:

* **Color Palettes:** Colormaps match the mathematical taxonomy of the underlying data. Bi-directional diverging structures (e.g., correlation datasets) map to diverging colormaps like `coolwarm` or `RdBu_r`. Uni-directional continuous patterns pull from sequential spaces like `viridis`, `plasma`, or `inferno`.
* **Typography Controls:** Fonts alternate between Sans-Serif options (Arial, DejaVu Sans, Liberation Sans) and Serif options (Times New Roman, Georgia) to ensure model generalization across typefaces.
* **Label Densities:** Label text rotation changes based on the density of the grid matrix. Axis text dynamically tilts to $0^\circ$ or $90^\circ$ to maintain readability and avoid character occlusion in dense layouts.

---

## 7. Recent Sources & Benchmarks

The programmatic conversion of data tables into synthetic charts is a key technique used to build datasets for training vision-language models. Recent research highlights the use of synthetic chart pipelines to address task complexity and chart reasoning challenges:

* **Zhou, M. et al. (ChartAct 2026).** *ChartAct: A Benchmark for Dynamic Chart Understanding*. arXiv preprint arXiv:2605.26994.
* Deploys dynamic multi-state rendering tracking to evaluate models on interactive chart tasks, highlighting the challenge of extracting hidden data values from visual interfaces.


* **Yang, Z. et al. (ChartMimic 2025).** *ChartMimic: Evaluating LMMs' Code Generation Elements via Visual Mimicking*. arXiv preprint arXiv:2501.03152.
* Uses chart-to-code pipelines to analyze how large multimodal models convert complex visual compositions into executable rendering code.


* **Rodriguez, J. et al. (StarVector 2025).** *From Charts to Code: A Hierarchical Benchmark for Multimodal Models*. OpenReview / Structural Vision Workshop.
* Explores chart code extraction by testing vector primitives and procedural code strings against complex multi-panel and multi-variable figures.


* **Han, S. et al. (EncQA 2026).** *EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts*. IEEE Transactions on Visualization and Computer Graphics.
* Establishes visual encoding benchmarks to evaluate model reasoning against chart features, including grid alignments and gradient structures.

Programmatic generation of synthetic pie charts relies on a specialized code-guided synthesis loop that maps compositional data vectors strictly to angular geometries on a 2D simplex canvas. This pipeline enforces topological consistency across all visual components, creating clean, annotated assets for machine learning models and multimodal benchmarks without human annotation error.

---

## 1. Pipeline Architecture & Domain Context

The pie chart synthesis engine models data bounded by a relative proportional ceiling ($100\%$ or a unit scale of $1.0$). The workspace isolates label taxonomies and statistical distributions across distinct application profiles:

* **Scientific Domain:** Simulates fractional breakdowns from quantitative assays such as chemical composition matrices, cellular population breakdowns in flow cytometry, or relative metagenomic abundances. Labels draw from exact taxonomic and molecular dictionaries (e.g., *Gene Symbol, Pathway, Metabolite Variant*).
* **Business Domain:** Emulates revenue segment tracking, regional market distributions, product category mixes, or customer churn cohort weights. Labels are mapped using organizational taxonomies (e.g., *Product SKU, Sales Region, Account Type*).

---

## 2. Compositional Data Models & Mathematical Formulations

Pie charts represent data on the **Aitchison Simplex** ($\mathcal{S}^D$), a restricted vector space where all elements must be strictly positive and sum exactly to 1:


$$\mathcal{S}^D = \left\{ \mathbf{x} = [x_1, x_2, \dots, x_D] \in \mathbb{R}^D \mid x_i > 0, \sum_{i=1}^D x_i = 1 \right\}$$

The algebraic operations and distributions used to generate and manipulate these compositions include the following methods:

### Simplex Operators

* **Closure ($\mathcal{C}$):** Projects any raw positive vector into the simplex space by dividing each component by the cumulative sum:

$$\mathcal{C}(\mathbf{x}) = \left[ \frac{x_1}{\sum_{k=1}^D x_k}, \frac{x_2}{\sum_{k=1}^D x_k}, \dots, \frac{x_D}{\sum_{k=1}^D x_k} \right]$$


* **Perturbation ($\oplus$):** Represents compositional addition on the simplex, performing component-wise multiplication followed by closure scaling:

$$\mathbf{x} \oplus \mathbf{y} = \mathcal{C}\left( [x_1 y_1, x_2 y_2, \dots, x_D y_D] \right)$$


* **Power Transform ($\odot$):** Represents compositional scalar multiplication, raising components to a power before projection:

$$\alpha \odot \mathbf{x} = \mathcal{C}\left( [x_1^\alpha, x_2^\alpha, \dots, x_D^\alpha] \right)$$


* **Aitchison Distance ($d_A$):** Computes the true log-ratio geometric distance between two compositions using the geometric mean $g(\mathbf{x}) = (\prod_{i=1}^D x_i)^{1/D}$:

$$d_A(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^D \left( \ln\frac{x_i}{g(\mathbf{x})} - \ln\frac{y_i}{g(\mathbf{y})} \right)^2}$$



### Sampling Distributions

The number of components is randomized between $D \in [3, 12]$. Slices are then generated using specialized probability distributions selected via a weighted configuration matrix:

* **Asymmetric Dirichlet Distribution:** Generates standard proportional vectors based on concentration parameters $\boldsymbol{\alpha}$:

$$f(\mathbf{x}; \boldsymbol{\alpha}) = \frac{1}{\mathrm{B}(\boldsymbol{\alpha})} \prod_{i=1}^D x_i^{\alpha_i - 1}$$



An asymmetry multiplier trend ($\text{trend} \in [1.0 + s, 1.0 - s]$ where $s \in [0.0, 1.5]$) can be applied to $\boldsymbol{\alpha}$ to systematically vary component sizes.
* **Pareto Proportional Shares:** Samples power-law distributions to replicate extreme disparities, such as market share dominance:

$$Y_i \sim \text{Pareto}(\alpha_{\text{shape}}, x_m) \quad \text{where} \quad \alpha_{\text{shape}} \in [1.16, 3.5], \quad \mathbf{x} = \mathcal{C}(\mathbf{Y})$$


* **Stick-Breaking Process:** Generates sequential decay proportions using continuous draws from a Beta distribution to construct decreasing structural tail sizes:

$$V_k \sim \text{Beta}(1, \gamma) \quad \text{where} \quad \gamma \in [1.0, 8.0]; \quad x_k = V_k \prod_{j=1}^{k-1} (1 - V_j)$$


* **Dominant Trace Model:** Splits the simplex into a few high-weight segments and a long tail of lower-weight components. A mixing parameter ($\phi \sim \text{Beta}(\alpha, \beta)$) defines the weight split between the dominant and trace vectors. Poisson read-depth sampling ($\mu = \mathbf{x} \cdot \text{depth}$) can be added to simulate counting noise from instruments.
* **Kummer-Dirichlet-Gaussian-Asymmetric (KDGA) Model:** Integrates a Kummer-Dirichlet exponential tilting factor into Gamma random variables to form an explicit asymmetric exponential density layout:

$$f(\mathbf{x}) \propto \left( \prod_{i=1}^D x_i^{\alpha_i - 1} \right) \exp\left( -\lambda \sum_{i=1}^D x_i \right)$$



### Composition Noise & Artifact Injections

* **Additive Logistic-Normal (ALN) Noise:** Perturbs the clean vector by mapping it to real space using an Additive Log-Ratio transform ($\text{alr}(\mathbf{x}) = [\ln(x_1/x_D), \dots, \ln(x_{D-1}/x_D)]$), adding a multivariate normal noise vector, and projecting back via the inverse transform:

$$\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \Sigma), \quad \Sigma = \mathbf{I} \cdot \sigma^2_{\text{variance}} \quad \text{where} \quad \sigma_{\text{variance}} \in [0.02, 0.08]$$


$$\mathbf{x}_{\text{noisy}} = \text{alr}^{-1}(\text{alr}(\mathbf{x}) + \mathbf{\epsilon})$$


* **Rounding Artifacts & Sum Deviations:** Simulates integer rounding constraints ($3$ decimal places). This operation often causes the rounded components to deviate from a perfect unit sum, creating small floating-point discrepancies:

$$\mathbf{x}_{\text{rounded}} = \text{round}(\mathbf{x}, d), \quad \delta = \sum_{i=1}^D x_{\text{rounded}, i} - 1.0$$



### Dynamic Entropy-Loss Tail Aggregation

When the number of slices exceeds a threshold ($\text{max\_slices} = 5$), low-weight tail slices are collapsed into a unified `"Other"` category. This consolidation is governed by Shannon entropy tracking to limit information loss:


$$H(\mathbf{x}) = -\sum_{i=1}^D x_i \log_2 x_i$$


The engine iteratively groups the smallest components, computing the resulting entropy reduction:


$$\Delta H = H(\mathbf{x}_{\text{original}}) - H(\mathbf{x}_{\text{aggregated}})$$


If $\Delta H$ exceeds a target threshold ($\text{entropy\_loss\_tolerance} = 0.5$), the consolidation stops and preserves the previous grouping layer to prevent excessive detail loss.

---

## 3. Visual Synthesis & Labeling Strategies

Slices are arranged chronologically on the canvas starting from a fixed top position ($startangle=90^\circ$). Slices can be ordered clockwise in descending magnitude, with any combined `"Other"` slice locked at the end of the sequence.

```
                     [ Pie Label Configuration Matrix ]
                                     |
     -----------------------------------------------------------------
    |                                |                                |
 [ default_leader ]         [ outside_pct_only ]            [ inside_pct_only ]
  Text labels placed         No text descriptions.           No text descriptions.
  outside; percentages       Percentages drawn               Percentages placed
  drawn inside patches.      outside via leader paths.       directly inside patches.

```

The pipeline alternates between four labeling strategies to vary the text layout:

1. **`default_leader`:** Places clear descriptive strings on the outside of the perimeter ($\text{labeldistance}=1.1$) and centers scalar percentage labels within the patch areas ($\text{pctdistance}=0.7$).
2. **`outside_pct_only`:** Suppresses descriptive strings entirely, drawing only numeric percentages outside the visual boundary ($\text{pctdistance}=0.8$).
3. **`inside_pct_only`:** Restricts all text to numeric percentages centered inside the wedge patches ($\text{pctdistance}=0.5$).
4. **`none`:** Suppresses all descriptive labels and percentage markers.

### Mechanical Separations & Paths

* **Wedge Explosion:** Slices can be separated radially using an explosion offset vector ($0.10$) applied to a single randomly selected slice.
* **Connector Primitives:** External labels can be linked to their corresponding patches using line paths (`pie_connector`). The path starts at the arc midpoint and extends to the center of the outer text label:

$$\mathbf{p}_{\text{start}} = \left[ x_c + r \cdot \cos(\theta_{\text{mid}}), \, y_c + r \cdot \sin(\theta_{\text{mid}}) \right]$$



---

## 4. Keypoint Geometry & YOLO Pose Mapping (5-Point Schema)

To support coordinate extraction and pose estimation benchmarks, the geometry of each slice is translated from raw coordinates to pixel bounds. The architecture implements a rigid **5-Keypoint Slice Boundary Schema** (`CLASS_MAP_PIE_POSE`):

```
                      1: ArcStart              2: ArcInter1
                         o--------------------o
                        /                      \
                       /                        \
                      /                          o 3: ArcInter2
                     /                            \
                    /                              \
                   o--------------------------------o
             0: WedgeCenter                     4: ArcEnd

```

### Coordinate Mapping Loop

1. **Center Verification:** The pipeline identifies the chart's origin $(0,0)$ and monitors the local wedge translation centers. For standard configurations, the center matches the origin; for exploded slices, the center shifts radially along the midpoint angle:

$$\mathbf{c}_{\text{wedge}} = [x_c, y_c] = [0.0 + \text{explode} \cdot \cos(\theta_{\text{mid}}), \, 0.0 + \text{explode} \cdot \sin(\theta_{\text{mid}})]$$


2. **Arc Boundary Tracking:** The boundaries of the sector are defined by the start angle ($\theta_1$) and end angle ($\theta_2$). Intermediate arc values are sampled at exactly one-third ($\theta_{\text{int1}}$) and two-thirds ($\theta_{\text{int2}}$) of the total angular span:

$$\theta_{\text{int1}} = \theta_1 + \frac{\theta_2 - \theta_1}{3.0}, \quad \theta_{\text{int2}} = \theta_1 + \frac{2(\theta_2 - \theta_1}{3.0})$$


3. **Keypoint Vector Composition:** The 5 keypoints are calculated in data coordinates using the wedge center and radius ($r$):
* $\mathbf{k}_0 = [x_c, y_c]$ (Wedge Center)
* $\mathbf{k}_1 = [x_c + r \cdot \cos(\theta_1), \, y_c + r \cdot \sin(\theta_1)]$ (Arc Start)
* $\mathbf{k}_2 = [x_c + r \cdot \cos(\theta_{\text{int1}}), \, y_c + r \cdot \sin(\theta_{\text{int1}})]$ (Arc Intermediate 1)
* $\mathbf{k}_3 = [x_c + r \cdot \cos(\theta_{\text{int2}}), \, y_c + r \cdot \sin(\theta_{\text{int2}})]$ (Arc Intermediate 2)
* $\mathbf{k}_4 = [x_c + r \cdot \cos(\theta_2), \, y_c + r \cdot \sin(\theta_2)]$ (Arc End)



These keypoints are transformed to image display pixels, flipped along the vertical axis ($y_{\text{pixel}} = h_{\text{img}} - y_{\text{data\_transformed}}$), and normalized to a $[0, 1]$ bounding box for export.

---

## 5. Themes, Aesthetics & Typography

Visual features change based on configured aesthetic style matrices:

* **Color Spaces:** The chart handles qualitative palettes differently depending on the chosen theme. The `excel` and `powerpoint` themes apply solid, high-saturation categorical colors, while the `ggplot` theme uses evenly spaced hue angles. The `default` or `seaborn_like` configurations apply smooth, continuous colormaps (e.g., `viridis`, `plasma`, `inferno`).
* **Typography Families:** Font classes vary randomly between Sans-Serif options (Arial, DejaVu Sans, Liberation Sans) and Serif options (Times New Roman, Georgia, DejaVu Serif) to improve OCR model generalization.
* **Legend Variations:** For dense compositions with more than 4 slices, labels are moved to an external legend block positioned adjacent to the chart bounding area ($\text{bbox\_to\_anchor}=(1.04, 0.5)$) to avoid text overlaps on small slices.

---

## 6. Recent Sources & Benchmarks

Programmatic synthesis pipelines are widely used in current computer vision and vision-language research to generate training data with precise ground-truth labels. Key publications and benchmarks reflecting these methodologies include:

* **Suh, S. et al. (Chart2Text-Enriched 2025).** *Chart2Text-Enriched: Towards Fine-Grained and Context-Aware Chart Summarization*. IEEE Transactions on Visualization and Computer Graphics.
* *Focuses on automated data extraction and structural analysis of complex multi-slice charts, evaluating models on their ability to interpret proportion ratios and legend groupings.*


* **Kondic, J. (ChartNet 2026).** *ChartNet: A million-scale, high-quality multimodal dataset for robust chart understanding*. arXiv preprint arXiv:2603.27064.
* *Demonstrates the use of large-scale programmatic synthesis loops to link data frames with visual primitives, providing a foundation for training models on multi-step chart reasoning.*


* **Liu, C. (ChartMaster 2026).** *ChartMaster: Boosting MLLMs for chart analysis through data, perception, and reasoning optimization*. OpenReview / International Conference on Learning Representations (ICLR).
* *Employs procedural rendering templates to convert raw data matrices into executable visualization strings, optimizing model training for chart understanding.*


* **Zhou, M. et al. (ChartAct 2026).** *ChartAct: A Benchmark for Dynamic Chart Understanding*. arXiv preprint arXiv:2605.26994.
* *Introduces dynamic chart interaction benchmarks, focusing on how vision models decode and extract proportion breakdowns from structural visual segments.*


* **Yang, Z. et al. (ChartMimic 2025).** *ChartMimic: Evaluating LMMs' Code Generation Elements via Visual Mimicking*. arXiv preprint arXiv:2501.03152.
* *Evaluates large multimodal models on chart code generation by tasking them with converting statistical figures into precise, renderable layout code.*

The programmatic generation of synthetic histogram charts without post-rendering image effects is implemented using an automated visualization architecture that connects statistical data distributions, mathematical binning constraints, and object layout tracking. This process enables the generation of pixel-perfect ground-truth coordinates used to train chart parsing and vision-language networks.

---

## 1. Pipeline Architecture & Domain Context

The histogram generation pipeline configures, alters, and compiles statistical observation sets before transforming them into graphic vector plots. The architecture isolates generation rules across explicit domains to ensure realistic representation:

* **Scientific Configuration:** Emulates continuous measurement arrays typical of real-world assays, such as fluorescence intensities in flow cytometry, spectroscopy profiles, or cell count distributions. It maps categorical parameters from a scientific pool (e.g., `Wavelength (nm)`, `Expression Level (a.u.)`, `Molarity (mM)`) and prioritizes highly skewed or zero-inflated probability density shapes.
* **Business Configuration:** Replicates metrics found in corporate analytics and economic dashboards, including product SKU lead times, sales performance frequencies, or order volume distributions. It maps taxonomies from commercial label indices (e.g., `Sales ($M)`, `Fulfillment Time (days)`, `Units Sold`).

The pipeline dynamically instantiates the sample workload density per chart, drawing a total number of samples uniformly between $N \in [300, 1200]$ observations.

---

## 2. Statistical Data Distributions & Mathematical Models

Rather than using uniform random generation, the data engine samples values from explicit probability distribution profiles defined in `_sample_histogram_distribution`. Each distribution is parameterized by a carry scale selector ($\text{scale} \in \{5, 10, 20, 50, 100\}$) to reflect authentic physical parameters:

### Standard Continuous Profiles

* **Normal (Gaussian) Distribution:** Represents standard symmetrical variations around a center value:

$$X_i \sim \mathcal{N}(\mu, \sigma^2) \quad \text{where} \quad \mu \in [-0.2, 0.8] \times \text{scale}, \quad \sigma \in [0.1, 0.35] \times \text{scale}$$


* **Gaussian Mixture Model (GMM):** Combines multiple distinct populations to build multimodal or clustered peaks:

$$f(x) = \sum_{k=1}^K w_k \cdot \frac{1}{\sigma_k \sqrt{2\pi}} \exp\left(-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right)$$



*Where the number of components $K \in [2, 4]$, mixture weights $\mathbf{w}$ are drawn from a uniform Dirichlet distribution, component means $\mu_k \in [-0.3, 1.0] \times \text{scale}$, and scales $\sigma_k \in [0.08, 0.35] \times \text{scale}$*.
* **Log-Normal Distribution:** Represents highly asymmetric, long-tailed positive measurement arrays:

$$\ln(X_i) \sim \mathcal{N}(\mu, \sigma^2) \quad \text{where} \quad \sigma \in [0.3, 0.9], \quad \mu = \ln(\text{scale} \times \mathcal{U}(0.2, 0.8))$$


* **Pareto Distribution:** Models power-law phenomena where data points compress near a low baseline threshold:

$$f(x) = \frac{b \cdot x_m^b}{x^{b+1}} \quad \text{where shape } b \in [1.1, 2.6], \quad \text{scale limit } x_m \in [0.3, 1.0] \times \text{scale}$$


* **Weibull Distribution:** Simulates equipment wear or lifetime constraints:

$$f(x) = \frac{c}{\lambda}\left(\frac{x}{\lambda}\right)^{c-1}\exp\left(-\left(\frac{x}{\lambda}\right)^c\right) \quad \text{where shape } c \in [0.8, 2.5], \quad \text{scale } \lambda \in [0.4, 1.1] \times \text{scale}$$


* **Gamma Distribution:** Simulates bounded, right-skewed physical values:

$$f(x) = \frac{x^{k-1}\exp\left(-\frac{x}{\theta}\right)}{\theta^k \Gamma(k)} \quad \text{where shape } k \in [1.2, 5.0], \quad \text{scale scale } \theta = \frac{\text{scale}}{k} \times \mathcal{U}(0.6, 1.2)$$


* **Beta Distribution:** Generates custom-shaped boundaries constrained within a specific metric window:

$$X_i = \text{scale} \times \text{Beta}(a, b) \quad \text{where shape parameters } a, b \in [0.6, 5.0]$$


* **Student's t-Distribution:** Replicates symmetric distributions with heavy tails to increase outlier incidence:

$$X_i \sim \text{t}(\nu, \mu, \sigma) \quad \text{where d.o.f. } \nu \in [2.5, 10.0], \quad \mu \in [-0.1, 0.4] \times \text{scale}, \quad \sigma \in [0.12, 0.35] \times \text{scale}$$


* **Truncated Normal Distribution:** Restricts data boundaries within strict limits to simulate controlled environments:

$$X_i \sim \text{TruncNorm}(\mu, \sigma^2, l, u) \quad \text{where lower limit } l = -0.2 \times \text{scale}, \quad \text{upper limit } u = 1.2 \times \text{scale}$$


* **Chi-Squared Distribution:** Simulates squared-error distribution characteristics:

$$X_i = \sigma \times \chi^2(k) \quad \text{where degrees of freedom } k \in [2.0, 10.0], \quad \sigma \in [0.2, 0.6] \times \text{scale}$$



### Zero-Inflated Mixture Distributions

To emulate realistic physical systems where specific sensors often register a absolute baseline measurement, the pipeline introduces zero-inflated models governed by a structural zero probability factor ($\pi \in [0.2, 0.6]$):

* **ZIP (Zero-Inflated Poisson):** Compiles count-based increments:

$$X_i = \begin{cases} 0 & \text{with probability } \pi \\ \text{Poisson}(\lambda) & \text{with probability } 1-\pi \end{cases} \quad \text{where rate } \lambda \in [2.0, 20.0]$$


* **ZINB (Zero-Inflated Negative Binomial):** Simulates overdispersed count variations:

$$X_i = \begin{cases} 0 & \text{with probability } \pi \\ \text{NegBinom}(n, p) & \text{with probability } 1-\pi \end{cases} \quad \text{where trials } n \in [1.5, 8.0], \, \text{prob } p \in [0.2, 0.8]$$


* **ZIG (Zero-Inflated Gaussian):** Combines structural zero records with a continuous normal curve profile:

$$X_i = \begin{cases} 0 & \text{with probability } \pi \\ \mathcal{N}(\mu, \sigma^2) & \text{with probability } 1-\pi \end{cases} \quad \text{where } \mu \in [0.2, 0.8] \times \text{scale}, \, \sigma \in [0.1, 0.35] \times \text{scale}$$



---

## 3. Data Degradation Filters

To challenge optical character recognition (OCR) and scene graph networks, the raw continuous arrays are passed through data degradation layers to simulate real-world noise:

* **Heteroscedastic Variance Floor:** Models experimental instrument measurement fluctuations where standard deviation scales dynamically according to data trends:
$$\text{Proportional Mode: } X'_i = X_i + \mathcal{N}\left(0, , (\gamma |X_i| + \delta \cdot \text{std})^2\right) \quad \text{where factor } \gamma \in [0.05, 0.15]$

$$\text{Exponential Mode: } X'_i = X_i + \mathcal{N}\left(0, \, (\delta \cdot \text{std} \cdot e^{\alpha |X_i|})^2\right) \quad \text{where growth factor } \alpha \in [0.02, 0.08]$$


* **Autocorrelation (Time-Series Dependency):** Applies an autoregressive $AR(1)$ filter layer to simulate data tracking dependency loops over sequential observations:

$$A_t = \phi A_{t-1} + \epsilon_t \quad \text{where lag multiplier } \phi \in [0.3, 0.8], \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$


* **Missingness Gaps:** Simulates random or structured data erasure faults spanning a total rate of $3\text{--}12\%$:
* *MCAR (Missing Completely at Random):* Drops elements uniformly across indices.
* *MAR (Missing at Random):* Missingness correlates with a secondary normalized covariate distribution scale.
* *MNAR (Missing Not at Random):* Selectively drops values falling past a targeted upper boundary quantile ($q \in [0.80, 0.95]$).


* **Outlier Generation Layers:** Modifies values at randomly selected coordinates to simulate equipment spikes or fat-finger typographical entry bugs:
* *Sensor Spikes:* Multiplies selected indices by an extreme magnitude scale ($5\text{--}25\times$) combined with a $30\%$ sign inversion rate.
* *Fat-Finger Errors:* Shifts decimal placements at targeted indices by applying integer multipliers across a discrete step array ($\times 0.01, \times 0.1, \times 10, \times 100, \times 1000$).



---

## 4. Dynamic Binning Strategies & Mathematical Optimization

The chart rendering boundaries are governed by a specialized helper object (`DynamicHistogramProcessor`). The algorithm selects an optimal interval compilation strategy based on sample length and data skewness:

```
                     [ Data Metric Structural Evaluation ]
                                       |
       -----------------------------------------------------------------
      |                                                                 |
 (Highly Skewed Distribution)                             (Stable Symmetric Data Profile)
  Evaluates Freedman-Diaconis or Doane                     Selects Sturges, Scott, or Auto
  interval calculations.                                  optimized partitioning grids.

```

### Friedman-Diaconis (FD) Estimator

Prioritizes outlier protection by utilizing the Interquartile Range ($\text{IQR} = Q_3 - Q_1$) to establish stable bin widths:


$$h_{\text{width}} = \frac{2 \cdot \text{IQR}(X)}{n^{1/3}}$$

$$\text{Total Bins } K = \left\lceil \frac{\max(X) - \min(X)}{h_{\text{width}}} \right\rceil$$

### Doane's Formula

Adjusts bin allocations when dealing with heavily non-normal or skewed distribution shapes by appending a skewness correction factor to Sturges' baseline rule:


$$K = 1 + \log_2(n) + \log_2\left(1 + \frac{|g_1|}{\sigma_{g1}}\right)$$


*Where $g_1$ represents the sample skewness value and $\sigma_{g1} = \sqrt{\frac{6(n-2)}{(n+1)(n+3)}}$ indicates the standard error coefficient*.

### Logarithmic Axis Intervals

When right-skewed profiles (e.g., Pareto or Log-Normal distributions) are active, the engine can implement logarithmic base-10 transformations to distribute bin widths evenly across orders of magnitude:


$$\text{Log\_Edges} = \text{linspace}\left(\log_{10}(\min(X)), \, \log_{10}(\max(X)), \, K + 1\right)$$

$$\text{Final Edges} = 10^{\text{Log\_Edges}}$$

---

## 5. Layout Compilation & Bounding Ground-Truth Capture

The graph rendering sequence utilizes Matplotlib's `ax.hist()` core layout engine, transforming the continuous data array into a sequence of discrete rectangular artists (`patches.Rectangle`):

```python
n, bins, patches = ax.hist(data, bins=edges, color=hist_color, zorder=2)

```

### Bounding Box Resolution Tracking

The pipeline parses these visual patches within the `get_granular_annotations` loop to capture precise object locations:

1. **Artist Traversal:** Isolates structural rectangles representing computed histogram bins.
2. **Display Boundary Evaluation:** Extracts absolute window bounds via `patch.get_window_extent(renderer)`, converting data-space corners into image pixel matrices ($[x_0, y_0, x_1, y_1]$).
3. **Visibility Quality Verification:** To prevent empty or clipped artifacts from cluttering the dataset, the pipeline applies a pixel-checking function (`has_non_background_pixels`). This ensures the patch region contains true color variations above a strict pixel standard deviation threshold (`PIXEL_STD_DEV_THRESHOLD = 10`) before exporting the normalized bounding labels (`CLASS_MAP_HISTOGRAM`).

### Contextual Text Annotations

* **Axis Titles:** Randomly draws semantic strings from structured parameter pools depending on the domain context. Y-axis labels default to statistical metrics such as `Frequency`, `Relative Frequency (%)`, `Probability Density`, or `Bin Count`.
* **Frequency Target Labels:** High-frequency intervals can feature precise text overlays centered above the bar caps ($10\%$ probability). Bars are sorted by frequency, and value tags are applied to the top $30\text{--}50\%$ of bins:

$$X_{\text{pos}} = x_{\text{bar}} + \frac{w_{\text{bar}}}{2}, \quad Y_{\text{pos}} = n_{\text{frequency}} + \Delta y_{\text{offset}}$$



---

## 6. Themes & Aesthetics Variations

The visual style of the generated histogram is governed by configurations defined in `themes.py`:

* **Color Spaces:** The color fill for bar patches is drawn from the theme's active palette configuration. The engine extracts categorical colors or samples continuous colormap midpoints (e.g., `viridis`, `plasma`, `inferno`) to define patch aesthetics.
* **Typography Controls:** Fonts alternate randomly across Sans-Serif options (`Arial`, `DejaVu Sans`, `Liberation Sans`) and Serif variations (`Times New Roman`, `Georgia`) to prevent training models from over-fitting to single typefaces.
* **Axis Rotations:** Tick labels tilt at angles of $0^\circ$, $45^\circ$, or $90^\circ$ when dense bin counts are active, preserving readability and avoiding character overlap.

---

## 7. Recent Sources & Benchmarks (2024–2026)

Code-guided chart synthesis is an active technique used in computer vision and multi-modal language research to construct high-quality benchmarks for chart interpretation and reasoning tasks:

* **Kondic, J. et al. (ChartGen 2025).** *ChartGen: Scaling Chart Understanding Via Code-Guided Synthetic Chart Generation*. arXiv preprint arXiv:2507.02424.
* *Details template-driven programmatic chart generation pipelines that map continuous data structures directly to renderable code primitives, creating large-scale benchmarks for scale tracking and layout parsing*.


* **Zhou, M. et al. (ChartAct 2026).** *ChartAct: A Benchmark for Dynamic Chart Understanding*. arXiv preprint arXiv:2605.26994.
* *Establishes benchmarks for interactive chart environments, evaluating model performance in interpreting distribution profiles, tracking grid coordinates, and extracting data from visual bins.*


* **Masry, A. et al. (ChartQAPro 2025).** *ChartQAPro: A New Benchmark for Complex Chart Question Answering*. Findings of the Association for Computational Linguistics (ACL).
* *Evaluates multimodal language models on statistical reasoning, using synthetic chart pipelines to test how models parse distribution shifts and variance markers.*


* **Han, S. et al. (EncQA 2026).** *EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts*. IEEE Transactions on Visualization and Computer Graphics.
* *Investigates visual encodings in charts, analyzing model capabilities in decoding visual features like bar heights, color scales, and interval bounds.*

The programmatic generation of synthetic scatter plot charts without post-rendering image effects relies on an automated synthesis framework that maps mathematical correlation models, coordinate space constraints, and visual properties directly onto a 2D rendering canvas. This approach enables the creation of pixel-perfect annotation tables used to train and benchmark multi-modal large language models (MLLMs) and object detection architectures.

---

## 1. Pipeline Architecture & Domain Context

The scatter plot synthesis pipeline configures vector lengths, structural relationships, and contextual layouts before compiling charts through Matplotlib. To match real-world publications, generation variables split across clear application domains:

* **Scientific Configuration:** Simulates data distributions typical of biological assays or physical experiments. Sample sizes ($N$) follow a prioritized distribution: $N \in [15, 20, 25, 30, 50, 75, 100]$ with relative sampling weights of $[0.15, 0.25, 0.20, 0.15, 0.15, 0.05, 0.05]$. This configuration uses label taxonomies from a technical pool (e.g., `Wavelength (nm)`, `Concentration (μM)`, `Absorbance (A.U.)`) and typically maps a single series configuration.
* **Business Configuration:** Replicates metrics found in corporate databases and economic intelligence panels. Sample sizes are larger to model denser transactional data: $N \in [50, 100, 200, 500]$ with weights $[0.20, 0.40, 0.30, 0.10]$. Labels draw from financial matrices (e.g., `Sales ($M)`, `Market Share (%)`, `Fulfillment Time (days)`) and often feature multi-series tracking layouts.

The global horizontal and vertical dimensions are mapped onto a standard carrying capacity scale ($\text{max\_scale}$) drawn uniformly from discrete value boundaries: $[50, 100, 200, 500, 1000]$.

---

## 2. Statistical Relationships & Mathematical Formulations

Rather than spreading coordinates uniformly across the figure area, the coordinate vectors ($\mathbf{x}, \mathbf{y} \in \mathbb{R}^N$) are synthesized through specific mathematical profiles. The statistical relationship type is selected using configured probabilistic weights: `strong_positive` (0.20), `moderate_positive` (0.25), `weak_positive` (0.20), `no_correlation` (0.15), `strong_negative` (0.05), `moderate_negative` (0.05), `nonlinear` (0.05), and `clustered` (0.05).

### Parametric Linear Correlations

For standard linear trends, the horizontal base vector $\mathbf{x}$ is sampled from independent continuous scales, sorted, and mapped against a targeted Coefficient of Determination ($R^2$) envelope:

1. **Target Bound Constraints:** * *Strong Trends:* $R^2 \sim \mathcal{U}(0.64, 0.81)$; Target Slope $|m| \in [0.5, 1.5]$.
* *Moderate Trends:* $R^2 \sim \mathcal{U}(0.36, 0.64)$; Target Slope $|m| \in [0.3, 0.8]$.
* *Weak Trends:* $R^2 \sim \mathcal{U}(0.09, 0.36)$; Target Slope $|m| \in [0.1, 0.5]$.
* *No Correlation:* $R^2 \sim \mathcal{U}(0.00, 0.09)$; Target Slope $m \in [-0.2, 0.2]$.


2. **Coordinate Matrix Mapping:** A clean target vector ($\mathbf{y}_{\text{perfect}}$) is computed relative to a random vertical intercept $c \sim \mathcal{U}(0.1, 0.3) \times \text{max\_scale}$:

$$\mathbf{y}_{\text{perfect}} = m \cdot (\mathbf{x} - \bar{x}) + c$$


3. **Targeted Noise Projection:** To satisfy the assigned $R^2$ boundary, the residual variance ($\sigma_{\text{noise}}^2$) is extracted from the perfect path vector variance ($\sigma_{\mathbf{y}_{\text{perfect}}}^2$) and injected as standard Gaussian noise:

$$\sigma_{\text{noise}}^2 = \sigma_{\mathbf{y}_{\text{perfect}}}^2 \cdot \left( \frac{1 - R^2}{R^2} \right)$$


$$\mathbf{y}_i = \mathbf{y}_{\text{perfect}, i} + \epsilon_i \quad \text{where} \quad \epsilon_i \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$$



### Nonlinear Matrix Formulations

When the `nonlinear` archetype is triggered, the pipeline samples from three curvilinear profiles, adding a proportional coefficient of variation noise layer ($\text{CV}_{\text{noise}} \in [0.08, 0.15]$) to scale error with point magnitude:

* **Quadratic Polynomial Profile:** $a \in [-0.002, 0.002]$, $b \in [-0.5, 0.5]$, and $c \sim \mathcal{U}(0.2, 0.5) \times \text{max\_scale}$:

$$\mathbf{y}_{\text{base}} = a\mathbf{x}^2 + b\mathbf{x} + c$$


* **Exponential Curve Profile:** $a \in [0.05, 0.15]$, $b \in [0.01, 0.03]$, and $c \sim \mathcal{U}(0, 0.2) \times \text{max\_scale}$:

$$\mathbf{y}_{\text{base}} = a \cdot \exp\left( \frac{b \cdot \mathbf{x}}{\text{max\_scale}} \right) + c$$


* **Logarithmic Transformed Profile:** $a \in [10, 30]$ and $c \sim \mathcal{U}(0, 0.3) \times \text{max\_scale}$:

$$\mathbf{y}_{\text{base}} = a \cdot \ln(\mathbf{x} + 1) + c$$


$$\text{Final Generation Layer: } \mathbf{y}_i = \mathbf{y}_{\text{base}, i} + \mathcal{N}\left(0, \, (\mathbf{y}_{\text{base}, i} \cdot \text{CV}_{\text{noise}})^2\right)$$



### Bivariate Multicluster Profiles (Elliptical Mixtures)

To challenge scene parsing networks, the `clustered` layout generates independent data groups ($K \in [2, 3, 4]$ clusters) using multivariate normal distributions to form dense visual clouds:


$$\text{Cluster Size } N_k = \max\left(5, \, \lfloor N/K \rfloor + \mathcal{U}\{-3, 3\}\right)$$


Each group $k$ is defined by a distinct cluster mean vector $\boldsymbol{\mu}_k = [\mu_{x,k}, \mu_{y,k}]^T$ sampled across central coordinates, and a bivariate elliptical covariance matrix $\Sigma_k$:


$$\Sigma_k = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{bmatrix}$$

$$\text{Where } \sigma_{xx}, \sigma_{yy} \sim \mathcal{U}(0.015, 0.08) \times \text{max\_scale}^2, \quad \sigma_{xy} = \rho \sqrt{\sigma_{xx} \sigma_{yy}} \quad (\rho \in [-0.5, 0.5])$$

$$\mathbf{P}_{i,k} \sim \mathcal{N}_2(\boldsymbol{\mu}_k, \Sigma_k)$$

---

## 3. Layout Rendering & Visual Parameter Mapping

The points are mapped onto the axes canvas using Matplotlib's `ax.scatter()` API, which translates data arrays into a collection of geometric path artists (`PathCollection`).

```
                  [ Scatter Dot-Workload Density Matrix ]
                                     |
     -----------------------------------------------------------------
    |                                |                                |
 (N < 30 Points)             (30 <= N < 100)                 (100 <= N < 500)
  High-intensity dots.        Medium weight markers.          Dense, thin markers.
  Size s ∈ [60, 100] pt        Size s ∈ [30, 60] pt            Size s ∈ [15, 30] pt

```

### Adaptive Dot Sizing Matrix

To ensure dots remain clear and legible across varying data densities, the marker area parameter ($s$, measured in points squared) scales inversely with the sample size workload $N$:

* $N < 30 \implies s \sim \mathcal{U}(60, 100)$
* $30 \le N < 100 \implies s \sim \mathcal{U}(30, 60)$
* $100 \le N < 500 \implies s \sim \mathcal{U}(15, 30)$
* $N \ge 500 \implies s \sim \mathcal{U}(5, 15)$

### Geometric Customization Fields

* **Marker Primitives:** Points iterate randomly across geometric paths: circles (`'o'`), squares (`'s'`), triangles (`'^'`), diamonds (`'D'`), or crosshairs (`'+'`).
* **Alpha Transparency Filters:** To manage overplotting in high-density regions, marker opacities are restricted between $\alpha \in [0.6, 0.8]$ to preserve visibility where points overlap.
* **Statistical Trendline Fits:** For linear trends (excluding `no_correlation` and `clustered` variations), a least-squares linear regression line is added with a $70\%$ activation probability. The pipeline fits the trend via `np.polyfit(x, y, 1)` and renders it across the dataset's span using a thin dashed stroke style (`'--'`).

---

## 4. Ground-Truth Geometric Extraction (Bounding Boxes & Keypoints)

During figure compilation, the pipeline extracts low-level graphical coordinates within the `get_granular_annotations` function to generate structured annotations for training object recognition models:

1. **Artist Target Selection:** The engine traverses the scene graph, identifies targets matching `isinstance(artist, PathCollection)`, and extracts data coordinates via `artist.get_offsets()`.
2. **Display Pixel Scale Transformation:** Coordinates are mapped from data space to absolute canvas pixel locations using the axis transformation matrix:

$$\mathbf{X}_{\text{pixel}}, \mathbf{Y}_{\text{pixel}} = \text{ax.transData.transform}([\mathbf{x}_i, \mathbf{y}_i])$$


3. **Marker Radius Resolution:** The physical point radius in display pixels is calculated from the marker area parameter $s$ to define precise bounding boundaries around each scatter center:

$$r_{\text{pixel}} = \sqrt{\frac{s}{\pi}}$$


4. **Normalized YOLO Formatting:** Bounding limits are computed for each point, normalized against the global canvas dimensions ($w_{\text{img}}, h_{\text{img}}$), and exported under a unified label schema (`CLASS_MAP_SCATTER`):

$$\text{BBox}_i = \left[ x_{\text{center}}, \, y_{\text{center}}, \, w, \, h \right] = \left[ \frac{\mathbf{X}_{\text{pixel}, i}}{w_{\text{img}}}, \, \frac{\mathbf{Y}_{\text{pixel}, i}}{h_{\text{img}}}, \, \frac{2r_{\text{pixel}}}{w_{\text{img}}}, \, \frac{2r_{\text{pixel}}}{h_{\text{img}}} \right]$$



---

## 5. Themes, Aesthetics & Typography Variations

Visual attributes are adjusted using configured styling profiles defined in `themes.py`:

* **Color Palettes:** Multi-series configurations assign distinct high-contrast qualitative hues (e.g., `tab10`, `Set2`) to keep categories separated. Single-series technical configurations can map point colors along continuous, perceptually uniform gradients (e.g., `viridis`, `plasma`, `coolwarm`) linked to density metrics or localized coordinate weights.
* **Axis Scales:** The coordinate grid can automatically switch from standard `linear` scaling to logarithmic (`log`) or symmetric-log (`symlog`) transformations based on the distribution span of the raw metrics.
* **Typography & Legend Positioning:** Fonts alternate systematically between Sans-Serif options (`Arial`, `DejaVu Sans`) and Serif typefaces (`Times New Roman`, `Georgia`) to improve model generalization. When multi-series legends are required, they are moved to external coordinates adjacent to the main chart grid ($\text{bbox\_to\_anchor}=(1.02, 0.5)$) to prevent text from overlapping data points.

---

## 6. Recent Sources & Benchmarks

Modern computer vision and vision-language benchmarks use programmatic, code-guided visualization engines to train and evaluate large models on complex chart reasoning and data extraction tasks:

* **Kondic, J. (ChartNet 2026).** *ChartNet: A million-scale, high-quality multimodal dataset for robust chart understanding*. arXiv preprint arXiv:2603.27064.
* *Introduces automated code-guided generation architectures that link structured data frames with visual primitives to build scale-tracking and chart-parsing datasets.*


* **Liu, C. (ChartMaster 2026).** *ChartMaster: Advancing Chart-to-Code Generation with Real-World Charts and Chart Similarity Reinforcement Learning*. OpenReview / International Conference on Learning Representations (ICLR).
* *Details template-driven chart generation pipelines that map numeric records to executable code modules, minimizing the visual domain gap between synthetic charts and real-world scientific figures.*


* **Zhou, M. et al. (ChartAct 2026).** *ChartAct: A Benchmark for Dynamic Chart Understanding*. arXiv preprint arXiv:2605.26994.
* *Establishes evaluation standards for dynamic and multi-panel visualization layouts, testing model capability in parsing coordinate systems and tracking scattered distributions.*


* **Masry, A. et al. (ChartQAPro 2025).** *ChartQAPro: A New Benchmark for Complex Chart Question Answering*. Findings of the Association for Computational Linguistics (ACL).
* *Evaluates multimodal models on complex statistical figures, using synthetic generation loops to analyze how models extract coordinate intersections and identify data clusters.*

Programmatic generation of synthetic area charts relies on an automated code-guided synthesis loop that maps multivariate time-series or continuous sequence vectors onto filled 2D geometric polygons. This pipeline enforces topological alignment between the underlying numerical tables and the generated graphical elements, producing high-fidelity ground truth for object detection, layout parsing, and structural pose estimation models without human annotation discrepancies.

---

## 1. Pipeline Architecture & Domain Context

The area chart synthesis pipeline maps continuous data streams across a uniform horizontal coordinate vector $X = [0, 1, \dots, N-1]$. The workspace dynamically bounds its structural dimensions using configured application profiles to ensure realistic visual composition:

* **Complexity Metrics:** Charts pull a randomized configuration containing $K \in [1, 4]$ concurrent data series tracks evaluated over a sequence length of $N \in [8, 25]$ discrete points. The global amplitude baseline is controlled by a scale threshold ($\text{max\_scale} \in \{50, 100, 500, 1000\}$).
* **Scientific Domain:** Models continuous physical properties, such as progressive spectral absorption bands, cumulative chromatography yields, or multi-replicate metabolic velocities over time. Labels are drawn from specialized technical vocabularies (e.g., `Wavelength (nm)`, `Concentration (μM)`, `Expression Level (a.u.)`).
* **Business Domain:** Emulates operational metrics, including cumulative SaaS revenue trajectories, stacked transaction volumes across regions, or cohort retention trends. Labels draw from financial matrices (e.g., `Sales ($M)`, `Fulfillment Time (days)`, `Units Sold`).

---

## 2. Mathematical Stacking Formulations & Models

The data rows ($\mathbf{y}_k \in \mathbb{R}^N$) for each series $k$ are initialized via explicit continuous equations (e.g., sigmoidal Hill responses, Michaelis-Menten kinetics, or multi-component seasonal ARIMA trends) before passing into a spatial stacking engine. The pipeline processes these lines using three distinct stacking modes:

```
[ Overlapping Mode ]           [ Stacked Mode ]            [ Percentage Mode ]
    Y_max = max_scale              Y_max = Σ(y_k)              Y_max = 100%
       
       / \                            / \                         /-----\
      /   \                          /   \                       |       |
     /=====\                        /=====\                      |=======|
    /       \                      /       \                     |       |
   -----------                    -----------                    -----------
   (Alpha = 0.5)                  (Alpha = 0.7)                  (Normalized Sum)

```

### Overlapping Area Mode

Each data series is evaluated independently over the full carrying capacity scale ($\text{series\_max} = \text{max\_scale}$). Polygons are filled directly from the absolute zero baseline coordinate:


$$\text{Baseline: } \mathbf{y}_{\text{start}, k} = \mathbf{0}, \quad \text{Boundary: } \mathbf{y}_{\text{end}, k} = \mathbf{y}_k$$


To maintain legibility across overlapping boundaries, the fill opacity is lowered to an alpha value of $\alpha = 0.5$.

### Stacked Area Mode

To prevent cumulative vertical overflows from clipping past the chart layout limits, individual tracks are scaled relative to the series volume:


$$\text{series\_max} = \frac{\text{max\_scale}}{\max(1, K)}$$


Polygons accumulate sequentially on top of the preceding data layers. The baseline coordinate vector for series $k$ matches the total cumulative height of all prior series tracks:


$$\mathbf{y}_{\text{start}, k} = \sum_{i=0}^{k-1} \mathbf{y}_i, \quad \mathbf{y}_{\text{end}, k} = \mathbf{y}_{\text{start}, k} + \mathbf{y}_k$$


This configuration uses a higher fill opacity ($\alpha = 0.7$) to establish clean, solid layers.

### Percentage / Normalized Area Mode

Ensures the total filled height equals a constant $100\%$ metric across all positions. Raw generated tracking rows ($\mathbf{y}_{\text{raw}, k}$) are passed through a normalization filter at each position index $j$:


$$\mathbf{y}_{k, j} = \left( \frac{\mathbf{y}_{\text{raw}, k, j}}{\sum_{i=0}^{K-1} \mathbf{y}_{\text{raw}, i, j} + \epsilon} \right) \times 100.0$$


*Where $\epsilon = 1\times10^{-6}$ prevents division-by-zero errors if all series register a zero value*. The layers are then stacked using the cumulative height approach, bounding the global axis ceiling to exactly $100$.

---

## 3. Structural Object Parsing & Annotation Specs

Area charts generate separate object tracking logs stored in two distinct annotation indices:

### Object Detection Paradigm (`CLASS_MAP_AREA_OBJ`)

Saves normalized YOLO format bounding box targets ($\left[x_{\text{center}}, y_{\text{center}}, w, h\right]$) for structural text and axis regions:

* `chart_title` (Class 3), `axis_title` (Class 1), `axis_labels` (Class 5), and `legend` (Class 2).
* Bbox limits map exactly to the window coordinates returned by the low-level rendering engine: `artist.get_window_extent(renderer)`.

### Pose Estimation Paradigm (`CLASS_MAP_AREA_POSE`)

Maps the filled boundary paths to a single target tracking class (`area_boundary`, Class 0) governed by a fixed **51-Keypoint Schema**:

$$\text{Keypoint Target Structure:} \quad \underbrace{0}_{\text{Start Point}} \rightarrow \underbrace{1\text{--}25}_{\text{Top Boundary Line}} \rightarrow \underbrace{26\text{--}49}_{\text{Bottom Boundary Line}} \rightarrow \underbrace{50}_{\text{End Point}}$$

The function `extract_area_pose_annotations_fixed` queries the polygon layers to extract the precise coordinate vertices ($\mathbf{x}_i, \mathbf{y}_i$) that form the perimeter boundaries. To match the fixed 51-point format across varying data path complexities, the vertices are processed using adaptive resampling filters:

```
                     [ Adaptive 51-Point Resampling Matrix ]
                                       |
         -------------------------------------------------------------
        |                                                             |
 [ Path Upsampling: N < 51 ]                        [ Path Downsampling: N > 51 ]
  Iterative Segment Splitting:                       Curvature-Based Pruning:
  - Finds the longest spatial path segment.          - Computes turn angles across vertices.
  - Splits it at the exact midpoint.                 - Keeps start and end points locked.
  - Loops until exactly 51 points are met.           - Discards straight or flat nodes first.

```

1. **Iterative Path Upsampling ($N < 51$):** Points are evaluated in structural sequence. The pipeline runs a heap queue to locate the longest spatial line segment, generates a new vertex at its midpoint, and recalculates the path lengths. This process repeats until the boundary path matches the 51-point requirement, distributing coordinates smoothly to avoid point clustering.
2. **Curvature Downsampling ($N > 51$):** Vertices are ranked by geometric importance by estimating the turn angles between adjacent segment vectors:

$$d^2y = y_{i+1} - 2y_i + y_{i-1}$$



The endpoints are locked, and low-scoring interior nodes with minimal directional variation are discarded until exactly 51 path coordinates remain.
3. **Coordinate Export:** The resampled path coordinates are translated into pixel coordinates, inverted along the vertical axis to match image space, and normalized into a $[0, 1]$ bounding box envelope:

$$x_{\text{norm}} = \frac{x_{\text{pixel}}}{w_{\text{img}}}, \quad y_{\text{norm}} = \frac{h_{\text{img}} - y_{\text{pixel}}}{h_{\text{img}}}$$



---

## 4. Themes, Aesthetics & Typography Variations

The visual properties of the area charts are adjusted dynamically using design rules configured in the thematic architecture:

* **Color Mapping:** The polygon fill colors match the active theme preset. The pipeline cycles through qualitative color lists (e.g., `tab10`, `Set2`) for distinct business groups or samples continuous colormaps (e.g., `viridis`, `plasma`) to represent ordered scientific sequences.
* **Structural Outlines:** To ensure clear visual distinction between stacked layers, a thin white border line ($1.5$ pt, $\alpha = 0.9$) is rendered at the top boundary of each area region.
* **Typography Controls:** Fonts alternate between Sans-Serif options (`Arial`, `DejaVu Sans`, `Liberation Sans`) and Serif variations (`Times New Roman`, `Georgia`) to prevent training models from over-fitting to single typefaces. Rotaion settings tilt text blocks ($0^\circ, 45^\circ, 90^\circ$) based on category label density.
* **Axis Limits Formatting:** Axis boundaries are adjusted based on the active stacking and scaling configurations. Percentage charts are locked to a strict $[0, 100]$ coordinate grid, while standard stacked charts apply an adaptive $15\%$ padding multiplier above the highest accumulated data vertex to prevent boundary clipping:

$$Y_{\text{limit\_top}} = \max\left(\mathbf{y}_{\text{stack}}\right) \times 1.15$$



---

## 5. Recent Sources & Benchmarks (2024–2026)

Code-guided chart synthesis is an active technique used in computer vision and multi-modal language research to construct high-quality datasets for chart interpretation and structural reasoning tasks:

* **Kondic, J. (ChartNet 2026).** *ChartNet: A million-scale, high-quality multimodal dataset for robust chart understanding*. arXiv preprint arXiv:2603.27064.
* *Details the implementation of program-driven chart generation pipelines that link structured data tables with visual primitives to build scale-tracking and boundary parsing benchmarks.*


* **Zhou, M. et al. (ChartAct 2026).** *ChartAct: A Benchmark for Dynamic Chart Understanding*. arXiv preprint arXiv:2605.26994.
* *Establishes evaluation standards for dynamic and multi-panel visualization layouts, testing model capabilities in parsing coordinate systems and tracking stacked area distributions.*


* **Yang, Z. et al. (ChartMimic 2025).** *ChartMimic: Evaluating LMMs' Code Generation Elements via Visual Mimicking*. arXiv preprint arXiv:2501.03152.
* *Uses chart-to-code pipelines to analyze how large multimodal models convert complex visual compositions, filled shapes, and chart legend coordinates into executable rendering code.*


* **Rodriguez, J. et al. (StarVector 2025).** *From Charts to Code: A Hierarchical Benchmark for Multimodal Models*. OpenReview / Structural Vision Workshop.
* *Explores chart code extraction by testing vector primitives, filled polygon boundaries, and procedural code strings against complex multi-panel and multi-variable figures.*


* **Han, S. et al. (EncQA 2026).** *EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts*. IEEE Transactions on Visualization and Computer Graphics.
* *Investigates visual encodings in charts, analyzing model capabilities in decoding visual features like layer heights, color scales, and filled region boundaries.*

In the context of programmatically generating datasets to train and evaluate multimodal large language models (MLLMs) and visual document processing systems, the synthesis pipeline goes far beyond basic rendering. To bridge the gap between idealized digital graphics and real-world documents, a comprehensive architecture of visual themes, context-aware semantic layers, and post-rendering degradation effects is deployed.

Below is an exhaustive description of the mechanics, formulas, statistical distributions, and architectural presets driving the themes and effects subsystems of this code-guided chart synthesis framework.

---

## 1. Thematic Framework & Aesthetic Engine

The design language of generated charts is controlled by a declarative engine that selects from pre-configured aesthetic palettes. Rather than producing uniform visual designs, the pipeline alternates between standard software packages, publication guidelines, and specialized accessibility standards.

### General Aesthetic Profiles

The global configuration repository initializes distinct `THEMES` settings, defining parameters for canvas background fill colors, grid configurations, typographic families, and border rules:

* **`default`:** A minimalist modern design featuring a clean white canvas, thin light-gray dashed grids ($0.8\,\text{pt}$ width), left and bottom spine alignments, and the `viridis` color space.
* **`excel`:** Emulates legacy corporate environments by introducing an explicit light-gray canvas background (`#F2F2F2`), heavy solid white grid lines ($1.5\,\text{pt}$ thickness), and a classic blue-accented axis profile (`#1F4E79`).
* **`ggplot`:** Replicates the signature design of the R visualization package, overlaying a dark-gray background mask (`#EBEBEB`) with solid white grid intervals while removing outer axis spines.
* **`prism`:** Tailored for medical and biological studies, this theme maintains a stark white backdrop, forces tick marks to point outward, and drops horizontal grid lines completely to keep focus on the data.
* **`retro`:** Implements an antique aesthetic utilizing a warm cream canvas backdrop (`#FFF8E7`), tan dotted grids (`#E6D6B5`), Georgia serif typography, and an earth-toned color scheme.
* **`accessible_compact` & `high_contrast_qualitative`:** Configured to comply with vision contrast guidelines, these profiles combine high-contrast qualitative palettes (e.g., `Set1` or custom Okabe-Ito compliance structures) with thick line widths ($2.0\,\text{pt}$) and larger minimum font sizes.

### Strict Publication Presets

To simulate real-world scientific data extraction tasks, the engine supports a subset of rigid configurations mapped directly to peer-reviewed journal templates (`PUBLICATION_THEMES`):

| Journal Preset ID | Baseline Canvas Dimensions (Inches) | Target Resolution (DPI) | Default Font Size | Spine Width | Line / Marker Properties |
| --- | --- | --- | --- | --- | --- |
| **`nature`** | $3.3 \times 2.4$ (Single Column) | $300$ | $7\,\text{pt}$ (Ticks/Labels) | $0.5\,\text{pt}$ | $0.5\,\text{pt}$ line / $3\,\text{pt}$ marker |
| **`science`** | $3.5 \times 2.5$ | $300$ | $6\,\text{pt}$ | $0.75\,\text{pt}$ | $0.75\,\text{pt}$ line / $3\,\text{pt}$ marker |
| **`cell`** | $6.5 \times 4.0$ | $300$ | $8\,\text{pt}$ | $0.6\,\text{pt}$ | $0.6\,\text{pt}$ line / $3\,\text{pt}$ marker |
| **`lancet`** | $7.0 \times 4.5$ | $300$ | $8\,\text{pt}$ (Times New Roman) | $0.8\,\text{pt}$ | Monochromatic / $0.8\,\text{pt}$ line |
| **`nanotech_short`** | $3.2 \times 2.4$ | $600$ | $7\,\text{pt}$ | $0.5\,\text{pt}$ | $0.6\,\text{pt}$ line / $2.5\,\text{pt}$ marker |

---

## 2. Context-Aware Domain Semantics

To ensure that text recognition models do not overfit to repetitive placeholder strings, labels are dynamically sampled from separate context-specific vocabularies (`SCIENTIFIC_DOMAIN_DICT` and `BUSINESS_DOMAIN_DICT`):

### The Scientific Vocabulary Space

* **Axis Label Indices:** Pulls from a curated index of over 100 professional measurement combinations (e.g., `Reads per million (RPM)`, `Binding Affinity (Kd, nM)`, `Optical Density (OD600)`, `Zeta Potential (mV)`, `NMR Chemical Shift (ppm)`).
* **Oncology & Pathway Matrices:** Features authentic biological entity sequences, such as common oncogenes (`TP53`, `BRCA1`, `BRCA2`, `EGFR`, `KRAS`) and cellular pathway networks (`PI3K-AKT`, `MAPK`, `WNT`, `Apoptosis`).

### The Corporate & Business Space

* **KPI Tracking Pools:** Maps enterprise metric taxonomies across horizontal and vertical planes, choosing combinations like `Monthly Active Users (MAU)`, `Customer Lifetime Value (LTV $)`, `Cost per Acquisition (CAC $)`, `Net Promoter Score (NPS)`, and `EBITDA ($)`.
* **Categorical Categorization Blocks:** Labels rows, columns, or legend segments using structural business records, including product codes, billing cycles, acquisition channels, and localized sales regions.

---

## 3. Post-Rendering Realism & Image Degradation Effects

Once the chart layout is compiled in vector format, the resulting image is passed through a sequence of data degradation filters implemented in `effects.py`. These filters introduce realistic imperfections, simulating printing errors, physical scanning artifacts, and compression noise.

### 1. Edge Color Normalization (`normalize_edgecolor`)

Ensures structural cohesion across bar boundaries or scatter points. If a distinct edge outline color is not defined, the module converts the internal patch color to HSV space, darkens the Value ($V$) channel, and converts it back to RGB to create a matching border tone:


$$H, S, V = \text{RGB\_to\_HSV}(R, G, B)$$

$$V_{\text{new}} = \max(0, \, V \times 0.6)$$

$$\text{Edge Color} = \text{HSV\_to\_RGB}(H, S, V_{\text{new}})$$

### 2. Additive White Gaussian Noise (`apply_noise_effect`)

Simulates sensory noise and grain fields introduced by physical scanners or imaging equipment. It maps a zero-mean normal distribution matching the array's shape and dimensions:


$$I_{\text{noisy}}(x, y, c) = \text{clip}\Big(I_{\text{raw}}(x, y, c) + \mathcal{N}(0, \sigma^2), \, 0, \, 255\Big) \quad \text{where} \quad \sigma \sim \mathcal{U}(2.0, 8.0)$$

### 3. Motion & Gaussian Blur (`apply_blur_effect`, `apply_motion_blur_effect`)

* **Gaussian Blur:** Simulates lens defocusing or low-quality scanning beds using a randomized blur radius ($\text{radius} \sim \mathcal{U}(0.5, 1.8)$).
* **Motion Blur:** Replicates hand-held document capture or scanning bed vibration. If a modern Pillow version is missing the native filter, the framework falls back to a custom directional convolution matrix. For a given pixel length size and transformation angle $\theta$, the kernel array sets its linear trajectory values to 1 and normalizes the sum:

$$x_{\text{index}} = \lfloor \text{center} + (i - \text{center}) \cdot \cos(\theta) \rfloor, \quad y_{\text{index}} = \lfloor \text{center} + (i - \text{center}) \cdot \sin(\theta) \rfloor$$


$$\mathbf{K}(y_{\text{index}}, x_{\text{index}}) = 1.0 \implies \mathbf{K}_{\text{normalized}} = \frac{\mathbf{K}}{\sum \mathbf{K}}$$



### 4. Pixelation, Low-Resolution, and Posterization Scales

* **`apply_low_res_effect`:** Simulates low bandwidth limits or camera downsampling. Images are downscaled using a bicubic algorithm and upscaled back using bilinear interpolation:

$$I_{\text{lowres}} = \text{Resize}_{\text{Bilinear}}\left(\text{Resize}_{\text{Bicubic}}(I, \, \text{scale}), \, \text{size}_{\text{original}}\right) \quad \text{where} \quad \text{scale} \sim \mathcal{U}(0.25, 0.60)$$


* **`apply_pixelation_effect`:** Simulates digital censorship or retro screen arrays by downscaling via bilinear filtering and immediately blowing the dimensions back up using a nearest-neighbor approach.
* **`apply_posterize_effect`:** Simulates color-palette restrictions or color quantization artifacts by converting the image space into an adaptive index profile containing a fixed number of colors ($N \in \{16, 32, 64\}$).

### 5. Specialized Text-Targeted Degradation (`apply_text_degradation_effect`)

To prevent deep reading networks from over-relying on high-contrast, perfectly rendered text regions, this module applies selective degradation to textual areas. It gathers title blocks, legend regions, and axis tick labels from the scene graph, isolates their bounding box screen regions via `get_window_extent`, and selectively applies distinct Gaussian blurs or pixelation filters exclusively to those text boxes.

### 6. Document Scanning Imperfections

* **Scanner Streaks (`apply_scanner_streaks_effect`):** Replicates physical scanning line artifacts caused by dust accumulation on the sensor bar. It draws a set of horizontal lines across random height coordinates with a low alpha opacity ($5\text{--}20$) and light value profiles ($200\text{--}255$).
* **Scan Rotation (`apply_scan_rotation_effect`):** Simulates document misalignment on a flatbed scanner by introducing a slight rotation angle ($\theta \sim \mathcal{U}(-1^\circ, 1^\circ)$). The canvas corners are automatically filled with the image's background border color to prevent edge clipping.

### 7. Uneven Vignette & Lighting Gradients

* **Vignette (`apply_vignette_effect`):** Replicates shadows and light falloff near lens borders using a quadratic distance equation:

$$r(x, y) = \sqrt{(x - x_{\text{center}})^2 + (y - y_{\text{center}})^2}, \quad r_{\text{max}} = \sqrt{x_{\text{center}}^2 + y_{\text{center}}^2}$$


$$V(x, y) = \text{clip}\left(1.0 - \left(\frac{r(x, y)}{r_{\text{max}}}\right)^2, \, 0.3, \, 1.0\right) \implies I_{\text{final}} = I \cdot V(x, y)$$


* **Uneven Lighting (`apply_uneven_lighting_effect`):** Replicates flash hot-spots or room lighting variations. It supports linear profiles (tiling a 1D gradient line across the canvas dimensions) or radial setups that subtract a scaled mask multiplier based on the distance from the center:

$$M_{\text{radial}}(x, y) = 1.0 - \left(\frac{\text{distance}(x, y)}{\max(\text{distance})}\right) \times \text{intensity} \quad \text{where} \quad \text{intensity} \in [0.0, 1.0]$$



### 8. Chromatic Aberration (`apply_chromatic_aberration_effect`)

Simulates lens alignment errors, which manifest as color fringes near edges with high contrast transitions. The algorithm splits the image into its constituent Red, Green, and Blue channels, applies spatial translation matrices to shift the Red and Blue layers in opposite directions, and re-merges the channels:


$$I_{\text{red}}(x, y) = I_{\text{red}}(x + \Delta x_r, \, y + \Delta y_r), \quad I_{\text{blue}}(x, y) = I_{\text{blue}}(x + \Delta x_b, \, y + \Delta y_b)$$

$$\text{Merged Output} = \text{ConcatChannels}\left(I_{\text{red}}, \, I_{\text{green}}, \, I_{\text{blue}}\right)$$

### 9. Contextual Overlay Layers

* **UI Chrome (`apply_ui_chrome_effect`):** Wraps the chart inside a simulated application interface, such as a desktop web browser tab, complete with a light toolbar and navigation dots (red, yellow, and green).
* **Watermark Overlays (`apply_watermark_effect`):** Places diagonal text stamps (e.g., `CONFIDENTIAL`, `SAMPLE`, `DRAFT`) with low visibility opacities ($4\text{--}12\%$) across the canvas center.
* **Mouse Cursor Injections (`apply_mouse_cursor_effect`):** Injects a vector mouse arrow primitive at a randomized coordinate, creating real-world noise for visual models.

---

## 4. Homography & Perspective Warping Mechanics

The most geometrically complex operation in the pipeline is the perspective transformation filter (`apply_perspective_warp_effect`), which simulates physical document capture angles.

```
   [ Source Vector Coordinates ]                [ Target Warped Coordinates ]
      (0,0)---------------(W-1,0)                    (rx0,ry0)---------\
        |                   |                            \              \-----(rx1,ry1)
        |                   |        ========>            \                    |
        |                   |      Transformation          \                   |
    (0,H-1)-----------(W-1,H-1)                             (rx3,ry3)---------(rx2,ry2)

```

The algorithm maps the 4 original corner coordinates of the canvas to randomized target coordinates shifted inward by a distortion factor:


$$\Delta x = \text{width} \times \text{distortion}, \quad \Delta y = \text{height} \times \text{distortion} \quad \text{where} \quad \text{distortion} = 0.08$$


The transformation maps point vectors from the original plane $[x, y]^T$ to the target warped projection $[u, v]^T$.

### Least-Squares Homography Solver

If the OpenCV package is unavailable, the pipeline falls back to an internal algebraic homography optimization routine (`_compute_homography`). The perspective projection can be expressed using a linear transformation matrix:


$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_1 & h_2 & h_3 \\ h_4 & h_5 & h_6 \\ h_7 & h_8 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

To solve for the 8 unknown parameters, the 4 corner mapping configurations are expanded into an aggregated system of linear equations ($A\mathbf{h} = \mathbf{b}$):


$$\begin{bmatrix} 
x_i & y_i & 1 & 0 & 0 & 0 & -u_i x_i & -u_i y_i \\ 
0 & 0 & 0 & x_i & y_i & 1 & -v_i x_i & -v_i y_i 
\end{bmatrix} \begin{bmatrix} h_1 \\ h_2 \\ h_3 \\ h_4 \\ h_5 \\ h_6 \\ h_7 \\ h_8 \end{bmatrix} = \begin{bmatrix} u_i \\ v_i \end{bmatrix}$$

The pipeline evaluates this system using standard least-squares estimation to compute the homography matrix:


$$\mathbf{h} = (A^T A)^{-1} A^T \mathbf{b}$$


The inverse homography matrix ($H^{-1}$) is passed to Pillow's transformation engine to map and interpolate pixels from the source canvas onto the warped target image.

---

## 5. Ground-Truth Bounding Box Adjustments

An important feature of this synthesis framework is that the degradation filters communicate with the bounding-box label generator. When operations like image clipping or canvas transformations alter the physical location of elements, the bounding box coordinates are adjusted accordingly:

* **Coordinate Shifts from Clipping (`apply_clipping_effect`):** When the clipping filter removes a slice along the top or left margins, the layout content shifts to pad the missing area. The module returns horizontal and vertical offset variables ($\Delta x, \Delta y$), which are added to the ground-truth text and object coordinates to keep the labels aligned:

$$x_{\text{adjusted}} = x_{\text{original}} + \Delta x, \quad y_{\text{adjusted}} = y_{\text{original}} + \Delta y$$


* **Perspective Adjustments via Homography:** For advanced perspective transformations, object boundary keypoints are multiplied by the computed homography matrix ($H$), projecting the data labels accurately into the new warped coordinate space:

$$\begin{bmatrix} x_{\text{warped}} \\ y_{\text{warped}} \\ w \end{bmatrix} = H \cdot \begin{bmatrix} x_{\text{original}} \\ y_{\text{original}} \\ 1 \end{bmatrix} \implies \mathbf{p}_{\text{final}} = \left[ \frac{x_{\text{warped}}}{w}, \, \frac{y_{\text{warped}}}{w} \right]$$



---

## 6. Recent Sources & Visual Degradation Benchmarks

This pipeline's strategy of pairing code-guided chart synthesis with image degradation matches current visual document processing and chart reasoning methodologies:

* **Zhou, M. et al. (ChartAct 2026).** *ChartAct: A Benchmark for Dynamic Chart Understanding*. arXiv preprint arXiv:2605.26994.
* *Explores multi-modal models in dynamic chart environments, focusing on how visual degradation filters affect data extraction and coordinate tracking.*


* **Yang, Z. et al. (ChartMimic 2025).** *ChartMimic: Evaluating LMMs' Code Generation Elements via Visual Mimicking*. arXiv preprint arXiv:2501.03152.
* *Uses advanced rendering engines to analyze large multimodal models on their ability to replicate complex chart styles, filled geometries, and legend coordinates.*


* **Rodriguez, J. et al. (StarVector 2025).** *From Charts to Code: A Hierarchical Benchmark for Multimodal Models*. OpenReview / Structural Vision Workshop.
* *Evaluates model performance in translating visual charts back into source code, testing robustness against document scanning artifacts and perspective warps.*


* **Han, S. et al. (EncQA 2026).** *EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts*. IEEE Transactions on Visualization and Computer Graphics.
* *Investigates visual encodings in charts, assessing model capabilities in decoding visual features like color gradients, line styles, and layout structures.*


* **Masry, A. et al. (ChartQAPro 2025).** *ChartQAPro: A New Benchmark for Complex Chart Question Answering*. Findings of the Association for Computational Linguistics (ACL).
* *Evaluates multimodal language models on statistical reasoning, using synthetic chart pipelines with integrated noise transformations to test model robustness against real-world document degradation.*