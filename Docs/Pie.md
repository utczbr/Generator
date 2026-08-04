Synthetic Compositional Data Generation and Aggregation Pipeline1. Compositional Data Spaces and Foundational TransformationsAitchison Simplex Definition: The sample space for compositional data sets consisting of relative information (proportions, probabilities, percentages) is the unit simplex $\mathcal{S}^{D} = \{ \mathbf{x} = \in \mathbb{R}^D \mid x_i > 0, \sum_{i=1}^D x_i = 1 \}$.Closure Operation: A mapping $C: \mathbb{R}^D_{+} \to \mathcal{S}^D$ to impose the unit-sum constraint.$C(\mathbf{x}) = \left$.Perturbation Operator ($\oplus$): The analogue of addition in the simplex space, used for ALN noise injection.$\mathbf{x} \oplus \mathbf{y} = C(x_1 y_1, \dots, x_D y_D)$.Power Transformation ($\otimes$): The analogue of scalar multiplication in the simplex space.$a \otimes \mathbf{x} = C(x_1^a, \dots, x_D^a)$.Aitchison Distance: Measures distance between compositions based on relative proportions rather than absolute Euclidean values.$d_A(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^D \left( \ln\frac{x_i}{g(\mathbf{x})} - \ln\frac{y_i}{g(\mathbf{y})} \right)^2}$.$g(\mathbf{x}) = \left( \prod_{i=1}^D x_i \right)^{1/D}$ (Geometric mean).2. Advanced Compositional Distributions2.1 Asymmetric and Hierarchical DistributionsStandard Dirichlet Distribution ($Di(\boldsymbol{\alpha})$):Multivariate generalization of the Beta distribution.Probability Density Function (PDF): $f(\mathbf{x} \mid \boldsymbol{\alpha}) = \frac{\Gamma\left(\sum_{i=1}^D \alpha_i\right)}{\prod_{i=1}^D \Gamma(\alpha_i)} \prod_{i=1}^D x_i^{\alpha_i - 1}$.Expected Value: $\mathbb{E}[x_i] = \frac{\alpha_i}{\sum_{j=1}^D \alpha_j}$.Variance: $\text{Var}(x_i) = \frac{\alpha_i (\alpha_0 - \alpha_i)}{\alpha_0^2 (\alpha_0 + 1)}$ where $\alpha_0 = \sum_{j=1}^D \alpha_j$.Covariance: $\text{Cov}(x_i, x_j) = \frac{-\alpha_i \alpha_j}{\alpha_0^2 (\alpha_0 + 1)}$.Asymmetry is achieved by setting $\alpha_i \neq \alpha_j$ to dictate varying expected category proportions.Kummer-Dirichlet Gamma (KDGa) Distribution:Provides high flexibility for modeling compositional data sets with outliers or complex diagnostic probability distributions.Incorporates an extra parameter $\lambda$ controlling dispersion and skewness, coupled with a gamma distribution generator.Normalizing Constant $C_2(\boldsymbol{\alpha}, \lambda)$:$C_2(\boldsymbol{\alpha}, \lambda) = \frac{\prod_{i=1}^p \Gamma(\alpha_i)\Gamma(\alpha_{p+1})}{\Gamma\left(\sum_{i=1}^p \alpha_i + \alpha_{p+1}\right)} {}_{1}F_{1}\left(\sum_{i=1}^p \alpha_i; \sum_{i=1}^p \alpha_i + \alpha_{p+1}; -\lambda\right)$.Where ${}_{1}F_{1}(\cdot; \cdot; \cdot)$ is the confluent hypergeometric function, $\alpha_i > 0$, and $-\infty < \lambda < \infty$.Probability Density Function (Baseline generator form):$h(\mathbf{x}) = C_2(\boldsymbol{\alpha}, \lambda) \left(1 - \sum_{i=1}^p G_i(x_i)\right)^{\alpha_{p+1} - 1} \prod_{i=1}^p g_i(x_i) G_i^{\alpha_i - 1}(x_i) \exp(-\lambda G_i(x_i))$.Where $g_i(\cdot)$ and $G_i(\cdot)$ are the PDF and CDF of the baseline distribution.Series Expansion Constant (using Pochhammer function $(\alpha)_n = \frac{\Gamma(\alpha+n)}{\Gamma(\alpha)}$):$C_2(\boldsymbol{\alpha}, \lambda) = \frac{\prod_{i=1}^p \Gamma(\alpha_i) \Gamma(\alpha_{p+1})}{\Gamma(\sum_{i=1}^p \alpha_i + \alpha_{p+1})} \sum_{m_1, \dots, m_p \ge 0}^\infty \frac{e^{-\lambda \sum_{i=1}^p m_i}}{\prod_{i=1}^p m_i!} \frac{\prod_{i=1}^p (\alpha_i)_{m_i}}{(\sum_{i=1}^p \alpha_i + \alpha_{p+1})_{\sum m_i}}$.Hierarchical Dirichlet Process (HDP) / Stick-Breaking Model:Allows for dynamic, theoretical unbounded components mapped to a fixed finite subset.Base distribution $H_0$, dispersion parameter $\gamma$.Stick length computation: $\beta_k \sim \text{Beta}(1, \gamma)$.Component weight computation: $w_k = \beta_k \prod_{i=1}^{k-1} (1 - \beta_i)$.Distribution ModelApplicationKey ParametersConstraintStandard DirichletGeneral baseline composition$\boldsymbol{\alpha} =$$\alpha_i > 0$KDGaHandling compositional outliers$\boldsymbol{\alpha}$, $\lambda$$-\infty < \lambda < \infty$Dirichlet ProcessUnknown/infinite categories$\gamma$, $H_0$$\sum_{i=1}^\infty w_i = 1$Pythonimport numpy as np
from scipy import stats
from scipy.special import gamma, hyp1f1, poch
from typing import Tuple, List, Union

class AdvancedDirichletSimulator:
    """
    Implements standard asymmetric Dirichlet, Kummer-Dirichlet Gamma (KDGa), 
    and Hierarchical Stick-Breaking Process for compositional simulation.
    """
    
    @staticmethod
    def asymmetric_dirichlet(
        n_samples: int, 
        alpha_params: np.ndarray
    ) -> np.ndarray:
        """
        Generates standard asymmetric Dirichlet variables.
        [1, 8]
        
        Parameters:
        n_samples (int): Number of independent compositional arrays to generate.
        alpha_params (np.ndarray): Asymmetric concentration parameters [a_1,..., a_n].
        
        Returns:
        np.ndarray: Array of shape (n_samples, len(alpha_params)) summing to 1 on axis 1.
        """
        if np.any(alpha_params <= 0):
            raise ValueError("All alpha parameters must be strictly greater than 0.")
        return stats.dirichlet.rvs(alpha_params, size=n_samples)
    
    @staticmethod
    def _kdga_normalizing_constant(alpha: np.ndarray, lambda_param: float) -> float:
        """
        Calculates the C2(alpha, lambda) normalizing constant for the Kummer-Dirichlet.
        
        """
        p = len(alpha) - 1
        alpha_p1 = alpha[-1]
        sum_alpha_p = np.sum(alpha[:-1])
        
        gamma_prod = np.prod([gamma(a) for a in alpha[:-1]]) * gamma(alpha_p1)
        gamma_sum = gamma(sum_alpha_p + alpha_p1)
        
        # Confluent hypergeometric function evaluation
        hypergeo = hyp1f1(sum_alpha_p, sum_alpha_p + alpha_p1, -lambda_param)
        
        return (gamma_prod / gamma_sum) * hypergeo

    @staticmethod
    def hierarchical_stick_breaking(
        n_samples: int, 
        n_components: int, 
        gamma_dispersion: float
    ) -> np.ndarray:
        """
        Implements the stick-breaking construction of the Dirichlet Process.
        
        
        Parameters:
        n_samples (int): Number of independent compositions.
        n_components (int): Truncation limit for the theoretically infinite process.
        gamma_dispersion (float): Concentration/dispersion parameter.
        
        Returns:
        np.ndarray: Matrix of shape (n_samples, n_components) summing to 1.
        """
        if gamma_dispersion <= 0:
            raise ValueError("Dispersion parameter gamma must be strictly positive.")
            
        betas = stats.beta.rvs(1, gamma_dispersion, size=(n_samples, n_components))
        # Enforce closure at the truncation limit
        betas[:, -1] = 1.0 
        
        weights = np.zeros_like(betas)
        weights[:, 0] = betas[:, 0]
        
        remaining_stick = 1.0 - betas[:, 0]
        for i in range(1, n_components):
            weights[:, i] = betas[:, i] * remaining_stick
            remaining_stick *= (1.0 - betas[:, i])
            
        return weights
2.2 Monopoly and Pareto Distributions (80/20 Rule Modeling)Vilfredo Pareto's Principle Context: Represents real-life phenomena such as wealth allocation, transaction volume, or business market share where ~80% of effects come from ~20% of causes.Pareto Probability Density Function (PDF):$f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}} \quad \text{for} \quad x \ge x_m$.Pareto Cumulative Distribution Function (CDF):$F(x) = 1 - \left( \frac{x_m}{x} \right)^\alpha$.Moments and Central Tendency:Expected Value: $\mathbb{E}[x] = \frac{\alpha x_m}{\alpha - 1}$ for $\alpha > 1$.Variance: $\text{Var}(x) = \frac{x_m^2 \alpha}{(\alpha - 1)^2 (\alpha - 2)}$ for $\alpha > 2$.Parameter Table & Real-World Calibration :α (Shape Parameter)Distribution CharacteristicsBusiness / Empirical Example$\alpha \approx 1.16$The exact mathematically derived 80/20 rule80% sales from 20% products; 80% bugs from 20% code$\alpha = 1.50$Very heavy tail; extreme values highly commonCity populations; massive market monopolies$\alpha = 2.50$Moderately heavy tailPersonal wealth distribution; broad market shares$\alpha = 3.50$Lighter tail; extreme values rareAdult height; fragmented market environmentsSimplex Mapping ($L_1$ Normalization):Standard Pareto samples $\mathbf{X} \in \mathbb{R}^D_{+}$ mapped to $\mathcal{S}^D$.$y_i = \frac{x_i}{\sum_{j=1}^D x_j}$.Pythonclass ParetoCompositionSimulator:
    """
    Simulates highly concentrated business market share distributions 
    using Pareto power-law distributions projected onto the Aitchison simplex.
    """
    
    @staticmethod
    def generate_pareto_shares(
        n_samples: int, 
        n_categories: int, 
        alpha_shape: float = 1.16,
        scale_min: float = 1.0
    ) -> np.ndarray:
        """
        Generates compositional market shares adhering to the 80/20 rule.
        [11, 14, 15]
        
        Parameters:
        n_samples (int): Number of market scenarios to simulate.
        n_categories (int): Number of competing entities/slices in the pie chart.
        alpha_shape (float): Shape parameter (1.16 approximates the 80/20 rule).
        scale_min (float): Minimum bound x_m for the Pareto distribution.
        
        Returns:
        np.ndarray: Compositional data bounded strictly between 0 and 1, summing to 1.
        """
        if alpha_shape <= 0:
            raise ValueError("Alpha shape parameter must be > 0.")
            
        # Draw from standard Pareto distribution
        pareto_raw = stats.pareto.rvs(
            b=alpha_shape, 
            scale=scale_min, 
            size=(n_samples, n_categories)
        )
        
        # Apply closure operation to project onto the simplex
        row_sums = pareto_raw.sum(axis=1, keepdims=True)
        compositions = pareto_raw / row_sums
        
        return compositions

    @staticmethod
    def verify_80_20_ratio(composition: np.ndarray) -> Tuple[float, float]:
        """
        Validates the concentration of a given compositional sample.
        Calculates the proportion of the total sum held by the top 20% of slices.
        """
        n_categories = composition.shape
        top_20_count = max(1, int(np.round(n_categories * 0.20)))
        
        sorted_comp = np.sort(composition)[::-1]
        top_20_sum = np.sum(sorted_comp[:top_20_count])
        
        return (top_20_count / n_categories), top_20_sum
2.3 Dominant-Trace Component ModelsContext: Microbiome datasets (e.g., Firmicutes vs. Bacteroidetes ), chemical compositions, or ecological networks where 1-2 major taxa dominate, leaving a massive tail of micro-slices.Bipartite Simplex Projection Framework:Set partition into Dominant components $\mathcal{D}$ (size $N_D$) and Trace components $\mathcal{T}$ (size $N_T$).Dominance scaling fraction: $\Phi \sim \text{Beta}(a, b)$, where $a \gg b$ to ensure $\mathbb{E}[\Phi] \to 1$.Dominant sub-composition generated via high-concentration Dirichlet: $\mathbf{x}_\mathcal{D} \sim \text{Dir}(\boldsymbol{\alpha}_\mathcal{D})$.Trace sub-composition generated via low-concentration Dirichlet: $\mathbf{x}_\mathcal{T} \sim \text{Dir}(\boldsymbol{\alpha}_\mathcal{T})$.Equation for final unified compositional vector $\mathbf{y}$:$y_i = \Phi \cdot x_{\mathcal{D}, i} \quad \forall i \in \mathcal{D}$$y_i = (1 - \Phi) \cdot x_{\mathcal{T}, i} \quad \forall i \in \mathcal{T}$Poisson Error Distribution Injection (Based on Generalized Linear Mixed-effects Models for sequence count data ):Read count approximation prior to closure: $c_i \sim \text{Poisson}(\mu_i = \lambda \cdot y_i)$.$\mathbf{y}_{\text{observed}} = C(\mathbf{c})$.Phylogenetic / Faith's PD Branch Length Equation :For ecological trace validity, unshared branch lengths can be mathematically defined as:
$\text{Fraction} = \frac{\sum_i^n b_i \times}{\sum_i^n b_i \times}$.Where $n$ is total branches, $b_i$ is branch length, $A_i, B_i$ are sequence descendent counts.Faith's PD: $PD_i = \sum_{j \in T} I_{ij} \cdot \text{branchlen}_j(T)$.Pythonclass DominantTraceSimulator:
    """
    Simulates chemical and microbiome compositional data featuring massive 
    dominant slices and heavy tails of micro-slices.
    """
    
    @staticmethod
    def simulate_microbiome_composition(
        n_samples: int,
        n_dominant: int = 2,    # e.g., Firmicutes and Bacteroidetes [16, 20]
        n_trace: int = 100,     # Micro-slices
        dominance_alpha: float = 90.0,
        dominance_beta: float = 10.0,
        poisson_read_depth: int = 10000
    ) -> np.ndarray:
        """
        Implements a bipartite simplex projection with optional Poisson count noise.
        [16, 17, 18, 20]
        
        Parameters:
        n_samples (int): Number of samples to generate.
        n_dominant (int): Number of massive dominant components.
        n_trace (int): Number of trace components.
        dominance_alpha (float): Beta distribution parameter pulling mass to dominant species.
        poisson_read_depth (int): Simulated sequencing depth for count-based noise.
        
        Returns:
        np.ndarray: Matrix of composition fractions.
        """
        # 1. Total dominance fraction (e.g., ~90% mass assigned to top 2 slices)
        phi = stats.beta.rvs(dominance_alpha, dominance_beta, size=(n_samples, 1))
        
        # 2. Dominant sub-composition (evenness controlled by high alpha)
        alpha_dom = np.full(n_dominant, 10.0)
        dom_comp = stats.dirichlet.rvs(alpha_dom, size=n_samples)
        
        # 3. Trace sub-composition (sparse, uneven noise controlled by low alpha)
        alpha_trace = np.full(n_trace, 0.05) 
        trace_comp = stats.dirichlet.rvs(alpha_trace, size=n_samples)
        
        # 4. Scale sub-compositions by phi and map to unified vector
        scaled_dom = dom_comp * phi
        scaled_trace = trace_comp * (1.0 - phi)
        true_composition = np.hstack((scaled_dom, scaled_trace))
        
        # 5. Apply Poisson read noise to simulate sequencer extraction 
        raw_counts = stats.poisson.rvs(mu=(true_composition * poisson_read_depth))
        
        # Prevent division by zero if read depth completely zeroed a sample
        row_sums = raw_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 
        
        observed_composition = raw_counts / row_sums
        return observed_composition
3. Realistic Data Degradation & Noise Injection3.1 Additive Logistic Normal (ALN) Noise ApplicationTheoretical Problem: The standard Dirichlet class is not closed under the basic simplex operations of perturbation. Applying standard Gaussian noise directly to compositional data violates the sum-to-1 constraint and produces hallucinations in predictive models.Log-Ratio Transformations:Move compositional data from simplex space to Euclidean space.Additive Log-Ratio (ALR): $\text{alr}(\mathbf{x}) = \left$.Centered Log-Ratio (CLR): $\text{clr}(\mathbf{x}) = \left$.Isometric Log-Ratio (ILR): Utilizes an orthonormal basis on the simplex. Transformation involves sequential binary partition trees.Additive Logistic Normal (ALN) Perturbation:Developed by Aitchison and Shen (1980) as a solution for multivariate normal variation on the simplex.Perturbation sequence modeled as: $\ln\left(\frac{x_{i}}{x_{D}}\right) = \ln\left(\frac{p_{i}}{p_{D}}\right) + \ln\left(\frac{\epsilon_{i}}{\epsilon_{D}}\right)$.Additive Central Limit Theorem logic: If perturbations are random, the ALR sum tends toward a multivariate normal pattern $L_{D-1}(\boldsymbol{\mu}, \Sigma)$.ALN Probability Density Function $f_{\mathbf{x}}(\mathbf{x})$:$f_{\mathbf{x}}(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^{D-1} \det(\Sigma)} \prod_{i=1}^D x_i} \exp\left\{ -\frac{1}{2} (\text{alr}(\mathbf{x}) - \boldsymbol{\mu})^T \Sigma^{-1} (\text{alr}(\mathbf{x}) - \boldsymbol{\mu}) \right\}$.Where $\boldsymbol{\mu}$ is a $(D-1)$ row vector, and $\Sigma$ is a positive definite square covariance matrix of order $D-1$.Simplex Injection via Inverse ALR ($alr^{-1}$):$\text{alr}^{-1}(\mathbf{y}) = C\left(\left\right)$.Perturbation operator $\oplus$ isolates strictly bounded noise: $\mathbf{x}_{\text{noisy}} = \mathbf{x} \oplus \boldsymbol{\epsilon} = C(x_1\epsilon_1, \dots, x_D\epsilon_D)$.Pythonclass AdditiveLogisticNormalInjector:
    """
    Applies mathematically sound ALN noise to compositional data 
    ensuring values remain strictly bounded and sum to 1 via Aitchison geometry.
    """
    
    @staticmethod
    def closure(x: np.ndarray) -> np.ndarray:
        """Applies closure operation C(x) to project vectors to the simplex."""
        return x / np.sum(x, axis=1, keepdims=True)
    
    @staticmethod
    def alr_transform(composition: np.ndarray) -> np.ndarray:
        """
        Computes the Additive Log-Ratio (ALR) transformation using the last component
        as the reference denominator.
        [4, 5, 26]
        """
        # Ensure strict positivity to prevent log(0)
        comp_safe = np.clip(composition, a_min=1e-12, a_max=None)
        reference_col = comp_safe[:, -1][:, np.newaxis]
        return np.log(comp_safe[:, :-1] / reference_col)

    @staticmethod
    def inverse_alr_transform(alr_data: np.ndarray) -> np.ndarray:
        """Computes the inverse ALR mapping back to the simplex."""
        n_samples = alr_data.shape
        exp_data = np.exp(alr_data)
        # Append the reference column (exp(0) = 1)
        reference_col = np.ones((n_samples, 1))
        reconstructed = np.hstack((exp_data, reference_col))
        return AdditiveLogisticNormalInjector.closure(reconstructed)

    @staticmethod
    def apply_aln_noise(
        compositions: np.ndarray, 
        variance_scale: float = 0.05
    ) -> np.ndarray:
        """
        Injects multivariate normal noise in the log-ratio space and maps back.
        
        
        Parameters:
        compositions (np.ndarray): Original simplex-bound data.
        variance_scale (float): Magnitude of the injected noise.
        
        Returns:
        np.ndarray: ALN-perturbed compositions summing to 1.
        """
        n_samples, n_components = compositions.shape
        
        # ALN operates in D-1 dimensional space
        d_minus_1 = n_components - 1
        cov_matrix = np.eye(d_minus_1) * variance_scale
        
        # Generate N(0, Sigma) noise
        log_noise = stats.multivariate_normal.rvs(
            mean=np.zeros(d_minus_1), 
            cov=cov_matrix, 
            size=n_samples
        )
        
        # Ensure log_noise is 2D even if d_minus_1 == 1
        if log_noise.ndim == 1:
            log_noise = log_noise[:, np.newaxis]
            
        alr_original = AdditiveLogisticNormalInjector.alr_transform(compositions)
        alr_noisy = alr_original + log_noise
        
        return AdditiveLogisticNormalInjector.inverse_alr_transform(alr_noisy)
3.2 Human Reporting Errors and Rounding ArtifactsMechanism of Reporting Errors: Human-reported survey data and legacy database systems rarely output mathematically perfect floating-point simplex arrays. Due to independent rounding of slices, the sum often diverges to percentages such as $99.9\%$ or $100.1\%$.Truncation and Rounding Function $R_k(x)$:To $k$ significant decimal places: $R_k(x_i) = \frac{\lfloor 10^k x_i + 0.5 \rfloor}{10^k}$.Summation Variance Equation:For mathematically pure compositions: $\sum_{i=1}^D x_i = 1.0$.For independently rounded subsets: $S = \sum_{i=1}^D R_k(x_i)$.Expected bounds of the arithmetic artifact: $S \in \{1 - 10^{-k}, 1, 1 + 10^{-k}\}$.When displayed as percentage integers ($x \times 100$): $P_i = R_3(x_i) \times 100 \implies \sum P_i \in \{99.9, 100.0, 100.1\}$.Pythonclass ReportingArtifactGenerator:
    """
    Simulates real-world reporting artifacts such as decimal rounding errors 
    that result in fractional sums slightly off from 1.0 (or 100%).
    """
    
    @staticmethod
    def inject_rounding_artifacts(
        compositions: np.ndarray, 
        decimals: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        
        
        Parameters:
        compositions (np.ndarray): Perfect simplex compositional arrays.
        decimals (int): Precision level (3 decimals = 1 decimal in percentage format).
        
        Returns:
        Tuple containing:
            - rounded_compositions: Matrix with human-like rounding.
            - sums: The actual row sums (revealing the 99.9% / 100.1% artifacts).
            - artifact_metrics: Dictionary containing counts of under/over-sums.
        """
        # Apply strict mathematical rounding independently to each component
        rounded_compositions = np.round(compositions, decimals)
        
        # Calculate resulting sums
        sums = np.sum(rounded_compositions, axis=1)
        
        # Define epsilon bounds for artifact detection based on precision
        epsilon = 10.0 ** (-decimals)
        
        mask_under = np.isclose(sums, 1.0 - epsilon)
        mask_over = np.isclose(sums, 1.0 + epsilon)
        mask_perfect = np.isclose(sums, 1.0)
        
        artifact_metrics = {
            "under_sum_count": int(np.sum(mask_under)),
            "over_sum_count": int(np.sum(mask_over)),
            "perfect_sum_count": int(np.sum(mask_perfect)),
            "max_deviation": float(np.max(np.abs(1.0 - sums)))
        }
        
        return rounded_compositions, sums, artifact_metrics
4. Intelligent Slice Management and Aggregation4.1 Entropy-Based Optimal ThresholdingThe Visualization Constraint: Pie charts display parts of a whole effectively only when the cognitive load is low. Empirical readability laws dictate a maximum of 5 to 6 slices. Data with a heavy tail of categories causes extreme visual clutter, requiring grouping into an "Other" category.Shannon Entropy ($H$) Formulation:Measures the information content and data dispersion (visual clutter).$H(\mathbf{x}) = -\sum_{i=1}^D x_i \log_2(x_i)$.Maximum entropy occurs when all slices are equal: $H_{\text{max}} = \log_2(D)$.Threshold Selection Optimization Model:Given sorted components in descending order: $x_{(1)} \ge x_{(2)} \ge \dots \ge x_{(D)}$.Define the aggregated mapping $A_k: \mathcal{S}^D \to \mathcal{S}^{k+1}$ which collapses the tail into a single index:$\mathbf{x}^{(k)} = \left$.Entropy of the dynamically aggregated composition:$H(\mathbf{x}^{(k)}) = -\sum_{i=1}^k x_{(i)} \log_2(x_{(i)}) - \left( \sum_{j=k+1}^D x_{(j)} \right) \log_2 \left( \sum_{j=k+1}^D x_{(j)} \right)$.Optimization Constraints:Hard Limit: $k+1 \le 6$ (Maximum visual slices).Tolerance Limit: Find minimum $k$ such that the information loss $\Delta H$ is acceptable:
$H(\mathbf{x}) - H(\mathbf{x}^{(k)}) \le \tau$.Aggregation CriterionFormula / RuleReferenceHard Count Threshold$\max(\text{slices}) \le 5$Dynamic Entropy Loss$H(\mathbf{x}) - H(\mathbf{x}^{(k)}) \le \tau$Flat Percentage CutoffGroup if $x_i < \text{threshold}$ (e.g., $1\%$)Pythonclass EntropyAggregator:
    """
    Dynamically calculates optimal thresholds for grouping minor components 
    into an 'Other' category based on visual readability, slice count, and dataset entropy.
    """
    
    @staticmethod
    def shannon_entropy(composition: np.ndarray) -> float:
        """
        Calculates Shannon Entropy in bits.
        [35, 36, 37]
        """
        # Exclude zeros to prevent log(0) NaN errors
        non_zero = composition[composition > 0]
        return -np.sum(non_zero * np.log2(non_zero))
        
    @staticmethod
    def dynamic_other_aggregation(
        composition: np.ndarray, 
        labels: List[str], 
        max_slices: int = 5,
        entropy_loss_tolerance: float = 0.5
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Iteratively collapses minor slices until entropy loss exceeds tolerance 
        or the hard slice-count limit is satisfied.
        [29, 30, 31, 32, 38]
        
        Parameters:
        composition (np.ndarray): Single 1D compositional array.
        labels (List[str]): Corresponding category names.
        max_slices (int): Absolute maximum slices permitted (including 'Other').
        entropy_loss_tolerance (float): Allowed drop in information bits.
        
        Returns:
        Tuple[np.ndarray, List[str]]: Aggregated composition and new labels.
        """
        if len(composition)!= len(labels):
            raise ValueError("Dimensions of composition and labels must match.")
            
        # Initial bounds check
        if len(composition) <= max_slices:
            return composition, labels
            
        # Sort arrays descending
        sort_idx = np.argsort(composition)[::-1]
        sorted_comp = composition[sort_idx]
        sorted_labels = [labels[i] for i in sort_idx]
        
        orig_entropy = EntropyAggregator.shannon_entropy(composition)
        
        # Test configurations from maximum allowable slices down to 2
        for k in range(max_slices - 1, 0, -1):
            
            # Collapse tail
            head_comp = sorted_comp[:k]
            tail_sum = np.sum(sorted_comp[k:])
            agg_comp = np.append(head_comp, tail_sum)
            
            agg_entropy = EntropyAggregator.shannon_entropy(agg_comp)
            entropy_loss = orig_entropy - agg_entropy
            
            # If information loss becomes too great, we must step back
            # However, if k == max_slices - 1, we MUST group to satisfy hard limits
            if entropy_loss > entropy_loss_tolerance and k < (max_slices - 1):
                # Revert to k+1 configuration
                k_prev = k + 1
                final_comp = np.append(sorted_comp[:k_prev], np.sum(sorted_comp[k_prev:]))
                final_labels = sorted_labels[:k_prev] + ['Other']
                return final_comp, final_labels
                
        # Fallback if loop completes (extremely skewed data)
        final_comp = np.append(sorted_comp[:max_slices-1], np.sum(sorted_comp[max_slices-1:]))
        final_labels = sorted_labels[:max_slices-1] + ['Other']
        return final_comp, final_labels
4.2 Readability-Optimized Sorting AlgorithmsVisual Layout Mechanics: The efficacy of a pie chart is deeply tied to its sorting topology. Randomly ordered data induces high cognitive load as observers attempt to estimate magnitude deltas visually.Algorithmic Rendering Rules:Origin Anchor: Rendering engine polar starting position must be locked at exactly 12 o'clock ($\theta_0 = \frac{\pi}{2}$ radians in standard trigonometry, but $\theta=0$ relative to pie plotting libraries).Magnitude Ascending/Descending: Values strictly sorted $x_1 \ge x_2 \ge \dots \ge x_k$.Clockwise Directionality: Angles project clockwise ($\theta_{i} < \theta_{i-1}$) rather than counter-clockwise.Absolute Index Positioning for "Other": The "Other" category circumvents magnitude sorting. It must strictly occupy the $k$-th position (final slice), bridging back to 12 o'clock.Polar Coordinate Translation Math:Total circle area mapping: $360^\circ$ or $2\pi$ radians.Angle delta for slice $i$: $\Delta\theta_i = 2\pi \cdot x_{(i)}$.Start angle: $\theta_{\text{start}}^{(i)} = 2\pi \sum_{j=1}^{i-1} x_{(j)}$.End angle: $\theta_{\text{end}}^{(i)} = 2\pi \sum_{j=1}^{i} x_{(j)}$.Computational Sorting Complexity:Merge Sort / Timsort (Python default argsort): Time complexity $\mathcal{O}(n \log n)$, Space complexity $\mathcal{O}(n)$.Required for dynamic arrays representing large entity counts (e.g., thousands of chemical trace components) prior to aggregation.Pythonclass PieChartSorter:
    """
    Optimizes array structures and generates polar coordinates for 
    maximum pie chart visual readability according to visualization best practices.
    """
    
    @staticmethod
    def sort_clockwise_descending(
        values: np.ndarray, 
        labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Sorts slices from largest to smallest, maintaining the 'Other' category 
        strictly at the final index regardless of its magnitude.
        [30, 32, 38, 42, 43, 44, 45]
        
        Parameters:
        values (np.ndarray): 1D array of composition values.
        labels (List[str]): List of corresponding labels.
        
        Returns:
        Tuple[np.ndarray, List[str]]: Ordered arrays ready for plotting libraries.
        """
        values_list = list(values)
        labels_list = list(labels)
        
        # 1. Identify and extract 'Other' if present to bypass magnitude sort
        has_other = 'Other' in labels_list
        if has_other:
            other_idx = labels_list.index('Other')
            other_val = values_list.pop(other_idx)
            labels_list.pop(other_idx)
            
        # 2. O(N log N) Timsort descending based on values
        sort_idx = np.argsort(values_list)[::-1]
        sorted_values = np.array(values_list)[sort_idx]
        sorted_labels = [labels_list[i] for i in sort_idx]
        
        # 3. Append 'Other' to the strict final index 
        if has_other:
            sorted_values = np.append(sorted_values, other_val)
            sorted_labels.append('Other')
            
        return sorted_values, sorted_labels
        
    @staticmethod
    def compute_polar_coordinates(sorted_values: np.ndarray) -> np.ndarray:
        """
        Maps the sorted compositional array directly to polar radian boundaries.
        
        
        Parameters:
        sorted_values (np.ndarray): 1D array of sorted composition values.
        
        Returns:
        np.ndarray: 2D array shape (N, 2) containing [theta_start, theta_end] per slice.
        """
        # Ensure pure sum to 1 before calculating geometry
        normalized = sorted_values / np.sum(sorted_values)
        
        # Convert proportions to radians
        angles = normalized * 2 * np.pi
        
        # Cumulative summation to find boundaries
        start_angles = np.cumsum(np.insert(angles, 0, 0))[:-1]
        end_angles = np.cumsum(angles)
        
        return np.column_stack((start_angles, end_angles))
