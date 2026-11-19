#set page(
  paper: "us-letter",
  margin: (x: 0.85in, y: 0.85in),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8.5pt, fill: rgb("#4b5563"))
      Overcoming High-Dimensional Collapse & Voronoi Overfitting in K-Means (`SID-KMeans`)
      #h(1fr)
      Andrew Rober (2026)
      #line(length: 100%, stroke: 0.4pt + rgb("#d1d5db"))
    ]
  }
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.1")

#let thmbox(title, body) = block(
  width: 100%,
  fill: rgb("#f8fafc"),
  stroke: (left: 2.5pt + rgb("#1d4ed8"), rest: 0.5pt + rgb("#cbd5e1")),
  inset: (x: 10pt, y: 8pt),
  radius: 2pt,
  breakable: true,
)[
  *#title.* #emph(body)
]

#let proofbox(body) = block(
  width: 100%,
  inset: (top: 2pt, bottom: 4pt),
)[
  _Proof._ #body #h(1fr) $square.filled$
]

#align(center)[
  #set par(justify: false)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(6pt)
  #text(size: 15.8pt, weight: "bold")[
    Overcoming High-Dimensional Collapse and Voronoi Overfitting in K-Means:\
    Spectral Intrinsic-Dimension Minimum Description Length (`SID-KMeans`)
  ]
  #v(6pt)
  #line(length: 100%, stroke: 0.8pt + black)
  #v(8pt)
  #text(size: 11pt, weight: "bold")[Andrew Rober]\
  #text(size: 9.5pt, fill: rgb("#374151"))[
    Independent AI & Statistical Machine Learning Research\
    `andrew_rober@hotmail.com` #sym.bullet `https://github.com/AndrewRober/IntrinsicDimKMeans`\
    November 2026
  ]
  #v(10pt)
]

#block(
  width: 100%,
  fill: rgb("#f3f4f6"),
  stroke: 0.5pt + rgb("#d1d5db"),
  inset: 11pt,
  radius: 3pt,
)[
  #align(center)[*Abstract*]
  #v(3pt)
  #set text(size: 9.25pt)
  Automatically determining the true number of clusters $K^*$ in $K$-Means and $K$-Means++ remains notoriously fragile in modern representation spaces ($d = 768–4096$). Within-Cluster Sum of Squares ($"WCSS"$) decreases monotonically with $K$; geometric validity indices (Silhouette, Caliński–Harabasz, Davies–Bouldin) cannot evaluate $K = 1$ and collapse onto $hat(K) = 2$ under hierarchical high-dimensional spectra; and Bayesian Information Criterion extensions (`X-Means`) exhibit a dual pathology in $RR^d$. Building upon, auditing, and rigorously reformulating our prior exploratory work @rober2023kmeans, we establish three theoretical results and introduce *Spectral Intrinsic-Dimension Minimum Description Length (`SID-KMeans`)*—a hyperparameter-free ($lambda$-free), scale- and rotation-invariant model selector:
  *(1) Voronoi Truncation Bias vs. Marginal Mixture Consistency (Theorem 1):* Hard $K$-Means partitioning reduces $"WCSS"$ even on a single unimodal Gaussian $cal(N)(bold(0), sigma^2 bold(I)_m)$ by $EE["WCSS"(2) / "WCSS"(1)] = 1 - frac(2, pi m)$, causing naive $"WCSS"$-BIC to diverge at rate $-frac(2, pi) N$ and overfit unconditionally ($100%$ false-positive rate). Restoring exact marginal `LogSumExp` mixture likelihood eliminates Voronoi overfitting while resolving softly touching clusters.
  *(2) Dual High-Dimensional Phase Transitions of Ambient BIC (Theorem 2):* When $K^*$ clusters lie in an $m$-dimensional subspace ($m lt.double d$), orthogonal noise $sigma_perp^2 > 0$ caps ambient log-distortion gains at $N Delta^2 / sigma_perp^2$ while parameter penalties grow as $Theta(K d ln N)$, forcing collapse to $hat(K) = 1$ above $d_("crit") = frac(N Delta^2, (K^* - 1) sigma_perp^2 ln N)$. Conversely, on smooth low-intrinsic-rank manifolds ($sigma_perp^2 approx 0$, $nu lt.double d$), dividing $"WCSS"$ by ambient $d$ inflates log-likelihood ratios by factor $d / nu gt.double 1$, causing runaway over-splitting ($hat(K) -> K_("max")$).
  *(3) Marchenko–Pastur + Satterthwaite Calibration (Theorem 3):* Combining Gavish–Donoho optimal singular-value bulk thresholding with Satterthwaite–Welch quadratic-form degrees of freedom $nu = ("Tr" hat(bold(Sigma)))^2 / "Tr"(hat(bold(Sigma))^2)$ calibrates squared Euclidean distances to their exact $chi_nu^2$ moment structure. Validated on an NVIDIA GB10 Grace Blackwell Superchip across 60 real `BAAI/bge-base-en-v1.5` (768D) and `BAAI/bge-large-en-v1.5` (1024D) document corpora, UCI benchmarks, and null-overfitting suites, `SID-KMeans` achieves *100% within-$plus.minus 1$ topic-count accuracy* (`MAE` $= 0.20–0.23$, vs. `0.53–0.70` for `X-Means`, `3.50` for Caliński–Harabasz, and `4.77–5.13` for the Gap Statistic) while running $4.6 times – 7.0 times$ faster than Gap.
]

#v(6pt)

= Introduction & Audit of Prior Heuristic Penalization

Partitioning $N$ observations $bold(X) = [bold(x)_1, dots, bold(x)_N]^top in RR^(N times d)$ into $K$ clusters by minimizing the Within-Cluster Sum of Squares ($"WCSS"(K) = sum_(k=1)^K sum_(bold(x)_i in C_k) norm(bold(x)_i - bold(mu)_k)_2^2$) via Lloyd's iterations @macqueen1967some @lloyd1982least and $K$-Means++ seeding @arthur2007kmeans is foundational across vector quantization, Retrieval-Augmented Generation (RAG) topic discovery, and representation analysis. Yet because $"WCSS"(K)$ decreases monotonically to zero at $K = N$, selecting $K^*$ without subjective visual "elbow" inspection requires an objective complexity penalty.

In an earlier exploratory project (`KMeansResearch`) @rober2023kmeans, we proposed balancing $"WCSS"$ against inter-centroid spread around the Virtual Centroid of Clusters ($"VCC" = frac(1, K) sum_(k=1)^K bold(mu)_k$), nearest-neighbor spacing $rho$, and an inverse-dimensionality factor $1 / d$ via a heuristic score $Delta = "WCSS" / (lambda rho d^(-1)) - (sum_k norm(bold(mu)_k - "VCC")_2 + 4 rho / lambda)$. Auditing that codebase and formulation uncovered four instructive mathematical and implementation defects:

1. *Coordinate-Space Mismatch (Figure 1B):* In the baseline C\# simulator, $"VCC"$ was retained in raw screen pixels ($approx 245$ px) while cluster centroids were normalized to $[0, 100]$ percentages prior to subtraction—causing $sum_k norm(bold(mu)_k - "VCC")_2$ to scale linearly with window pixel offset ($approx 275 K$) and creating an accidental visual crossing point with $"WCSS"$!
2. *Accidental Metric Cancellation & Seeding Mismatch:* The implementation read per-cluster mean squared distortion $"WCSS"_k / n_k$ rather than nearest-neighbor spacing $rho$, inadvertently canceling $"WCSS"$ from the numerator; moreover, centroid initialization used deterministic Gonzalez farthest-first traversal rather than randomized $D(bold(x))^2$ $K$-Means++ sampling @arthur2007kmeans.
3. *Broken Scale and Sample-Size ($N$) Invariance:* Because $"WCSS"(K) = Theta(N L^2)$ carries squared length units and scales linearly with sample count $N$, whereas $sum_k norm(bold(mu)_k - "VCC")_2 = Theta(K L)$ is unsquared and independent of $N$, no fixed tuning constant $lambda$ can remain invariant across coordinate rescalings $bold(X) mapsto alpha bold(X)$ or dataset sizes $N$.
4. *High-Dimensional Breakdown of Classical Replacements:* Crucially, even standard textbook replacements—Pelleg & Moore's `X-Means` @pelleg2000xmeans, Silhouette @rousseeuw1987silhouettes, Caliński–Harabasz @calinski1974dendrite, and the Gap Statistic @tibshirani2001estimating—fail severely on modern 768D–1024D Transformer embeddings, where semantic clusters occupy a curved, anisotropic low-intrinsic-dimension manifold ($nu lt.double d$).

#figure(
  image("../figures/fig4_spectrum_and_csharp_audit.png", width: 96%),
  caption: [*(A)* Singular energy spectrum $s_j^2$ of 6-topic `20 Newsgroups` embedded in `BAAI/bge-large-en-v1.5` ($d = 1024$): Gavish–Donoho bulk cutoff retains $m_("GD") = 105$ axes while exact Satterthwaite degrees of freedom equal $nu = 31.8 lt.double 1024$ (preventing a $32 times$ ambient likelihood inflation). *(B)* Audit of the pixel-vs-percentage coordinate warp fixed in our upgraded C\# simulator.]
)

= Exact Variance Geometry & Voronoi Overfitting (Theorem 1)

By Huygens' theorem, total centered dispersion $"TSS" := sum_(i=1)^N norm(bold(x)_i - macron(bold(x)))_2^2$ decomposes homogeneously in $L^2$ units as $"TSS" = "WCSS"(K) + "BCSS"(K)$, where $"BCSS"(K) = sum_(k=1)^K n_k norm(bold(mu)_k - macron(bold(x)))_2^2$ weights each centroid's squared displacement from the weighted global mean $macron(bold(x))$ by its population $n_k$ (the exact scale-consistent counterpart to @rober2023kmeans's unweighted $"VCC"$ sum).

Why does simply adding a parameter penalty $p_K ln N$ to $N m ln("WCSS"(K) / (N m))$—a widespread "Naive $"WCSS"$-BIC" heuristic—catastrophically overfit unimodal data? Because hard Voronoi tessellation truncates Gaussian tails even when *only one cluster exists*.

#thmbox("Theorem 1 (Voronoi Truncation Bias vs. Marginal Mixture Unimodal Consistency)")[
  Let $bold(x)_1, dots, bold(x)_N in RR^m$ be drawn i.i.d. from a single unimodal isotropic Gaussian $cal(N)(bold(0), sigma^2 bold(I)_m)$ ($K^* = 1$).
  1. *Asymptotic Failure of Naive $"WCSS"$-BIC:* Under optimal $K = 2$ hyperplane $K$-Means partitioning, half-space centroids converge almost surely to $bold(mu)_(1,2) = plus.minus sigma sqrt(2 / pi) bold(u)$ ($norm(bold(u))_2 = 1$), yielding
     $ lim_(N -> infinity) frac("WCSS"(2), "WCSS"(1)) = 1 - frac(2, pi m). $
     Consequently, the deviance drop satisfies $N m ln("WCSS"(2) / "WCSS"(1)) = N m ln(1 - frac(2, pi m)) + o_PP (N) <= -frac(2, pi) N approx -0.6366 N$, overwhelming any $O(m ln N)$ BIC penalty and causing $100%$ asymptotic overfitting ($hat(K) > 1$).
  2. *Assignment Entropy & Exact Marginal Mixture Restoration:* Encoding hard cluster assignments adds the Shannon code-length $+2 sum_(k=1)^K n_k ln(N / n_k) = +2 N H(hat(bold(pi)))$, shifting the per-sample asymptotic slope on $cal(N)(bold(0), sigma^2 bold(I)_m)$ at $K = 2$ to
     $ m ln(1 - frac(2, pi m)) + 2 ln 2 >= ln(1 - frac(2, pi)) + 2 ln 2 approx +0.3740 > 0 quad "for all " m >= 1. $
     Furthermore, evaluating the exact marginal Gaussian mixture deviance $-2 ell_("mix")(K) = -2 sum_(i=1)^N ln sum_(k=1)^K hat(pi)_k cal(N)(bold(x)_i | bold(mu)_k, hat(sigma)_K^2 bold(I)_m)$ via `LogSumExp` satisfies the exact Bayes identity
     $ -2 ell_("mix")(K) = -2 ell_("complete")(K) - 2 sum_(i=1)^N ln(1 + sum_(j != c_i) frac(hat(pi)_j, hat(pi)_(c_i)) exp(-frac(norm(bold(x)_i - bold(mu)_j)_2^2 - norm(bold(x)_i - bold(mu)_(c_i))_2^2, 2 hat(sigma)_K^2))), $
     preserving unimodal consistency ($K^* = 1$) while avoiding hard-ICL under-splitting @biernacki2000assessing on softly touching clusters.
]

#proofbox[
  By rotational symmetry of $cal(N)(bold(0), sigma^2 bold(I)_m)$, any hyperplane through the origin splits $RR^m$ into half-spaces along $z_1 = bold(u)^top bold(x) tilde.op cal(N)(0, sigma^2)$ with $EE[|z_1|] = sigma sqrt(2 / pi)$ and $"Var"(|z_1|) = sigma^2 (1 - 2 / pi)$, while the $(m - 1)$ orthogonal coordinates retain variance $sigma^2$. Summing variances gives $EE["WCSS"(2)] / (N m sigma^2) = ((m - 1) + (1 - 2 / pi)) / m = 1 - 2 / (pi m)$. Since $ln(1 - u) < -u$ for $u in (0, 1)$, $m ln(1 - 2 / (pi m)) < -2 / pi < 0$, proving $-Theta(N)$ divergence of Naive $"WCSS"$-BIC. Adding $2 H(1/2, 1/2) = 2 ln 2$ yields $g(m) := m ln(1 - 2 / (pi m)) + 2 ln 2$, which is strictly increasing on $m >= 1$ from $g(1) = +0.3740$ to $lim_(m -> infinity) g(m) = 2 ln 2 - 2 / pi = +0.7497 > 0$. The Bayes identity factors the largest exponent term $c_i = arg min_k norm(bold(x)_i - bold(mu)_k)_2$ out of `LogSumExp`.
]

= Dual High-Dimensional Pathologies & `SID-KMeans` (Theorems 2 & 3)

Even with exact assignment/mixture likelihoods, evaluating criteria directly in ambient $RR^d$ ($d gt.double 1$) fails via *two opposite mechanisms* depending on the ambient noise spectrum.

#thmbox("Theorem 2 (Dual High-Dimensional Failure Modes: Dilution Collapse vs. Collinearity Inflation)")[
  Let $K^* >= 2$ clusters have between-cluster signal $"BCSS"(K^*) = N Delta^2 > 0$ supported on an $m$-dimensional subspace $cal(S) subset RR^d$ ($m lt.double d$) with within-subspace variance $sigma_parallel^2$.
  1. *Regime A—Isotropic Ambient Noise Dilution ($sigma_perp^2 > 0$ on $cal(S)^perp$):* As $d -> infinity$ with fixed $(N, K^*, Delta^2, sigma_perp^2)$, $"WCSS"(K) = "WCSS"_(cal(S))(K) + N(d - m) sigma_perp^2$, so the ambient log-distortion gain hits a finite ceiling:
     $ lim_(d -> infinity) N d ln(frac("WCSS"(1), "WCSS"(K^*))) = frac(N Delta^2, sigma_perp^2) < infinity, $
     whereas the ambient BIC penalty $(K^* - 1)(d + 1) ln N -> +infinity$. Thus ambient BIC / `X-Means` collapses to $hat(K) = 1$ above the exact critical dimension
     $ d_("crit") = frac(N Delta^2, (K^* - 1) sigma_perp^2 ln N) + O(1). $
     Simultaneously, in Caliński–Harabasz $"CH"(K) = frac("BCSS"(K)/(K-1), "WCSS"(K)/(N-K))$, the denominator is dominated by $d sigma_perp^2$, reducing maximization to $arg max_(K >= 2) "BCSS"(K)/(K-1)$, which collapses onto $hat(K) = 2$ whenever the top between-cluster eigenvalue strictly exceeds the average of subsequent eigenvalues.
  2. *Regime B—Low-Rank / Anisotropic Manifold Inflation ($sigma_perp^2 = 0$, effective rank $nu lt.double d$):* When embeddings lie on an anisotropic manifold of effective rank $nu lt.double d$ with negligible orthogonal variance, $"WCSS"(K^*) / "WCSS"(1)$ remains invariant to $d$, so multiplying $ln("WCSS"(K)/"WCSS"(1))$ by ambient $d$ inflates effective log-likelihood ratios by factor $d / nu gt.double 1$, causing runaway over-splitting ($hat(K) -> K_("max")$).
]

#proofbox[
  In Regime A, $"WCSS"(1) / "WCSS"(K^*) = 1 + Delta^2 / (m sigma_parallel^2 + (d - m) sigma_perp^2)$. Using $ln(1 + epsilon) = epsilon + O(epsilon^2)$ as $d -> infinity$ gives $N d ln("WCSS"(1) / "WCSS"(K^*)) = N d Delta^2 / ((d - m) sigma_perp^2) + O(d^(-1)) -> N Delta^2 / sigma_perp^2$; equating this ceiling to $(K^* - 1) d ln N$ yields $d_("crit")$ (verified symbolically in SymPy). For $"CH"(K)$, $"WCSS"(K)/(N - K) = d sigma_perp^2 (1 + O(d^(-1)))$ is asymptotically constant in $K$; since $"BCSS"(2)/1 = lambda_1 ("between")$ while $"BCSS"(K)/(K-1) <= frac(1, K-1) sum_(j=1)^(K-1) lambda_j ("between") < lambda_1 ("between")$ for strictly decaying between-cluster spectra, $"CH"(K)$ peaks at $K = 2$. In Regime B, padding $(d - r)$ zero-variance coordinates leaves $"WCSS"(K)$ unchanged while multiplying $hat(sigma)_K^2 = "WCSS"(K)/(N d)$ by $1/d$, scaling deviance differences by $d / nu$.
]

To resolve *both* regimes without any user-tuned parameter $lambda$, `SID-KMeans` executes two spectral calibration steps on centered data $tilde(bold(X)) = bold(U) bold(S) bold(V)^top$:
1. *Marchenko–Pastur Bulk Thresholding (Curing Regime A):* When $min(N - 1, d) >= 10$, singular values below the Gavish–Donoho optimal hard threshold @gavish2014optimal $tau_("GD") = omega(beta) s_("med")$ ($beta = min(N, d)/max(N, d)$, $omega(beta) = 0.56 beta^3 - 0.95 beta^2 + 1.82 beta + 1.43$) arise from isotropic bulk noise and are truncated, projecting onto $tilde(bold(X))_("clean") = bold(U)_(s > tau_("GD")) bold(S)_(s > tau_("GD"))$ (if no singular value exceeds $tau_("GD")$, $hat(K) = 1$ immediately).
2. *Satterthwaite Effective Degrees of Freedom (Curing Regime B):* Even among retained signal singular values, neural embeddings exhibit steep power-law decay (Figure 1A). Theorem 3 calibrates the quadratic distance form to its exact $chi_nu^2$ moment structure.

#thmbox("Theorem 3 (Satterthwaite Quadratic-Form Calibration & SID-KMeans Objective)")[
  For any anisotropic Gaussian residual $bold(epsilon.alt) tilde.op cal(N)(bold(0), bold(Sigma))$ with eigenvalues $lambda_1 >= dots >= lambda_m >= 0$, the squared Euclidean distance $Q = norm(bold(epsilon.alt))_2^2 = sum_(j=1)^m lambda_j Z_j^2$ ($Z_j tilde.op cal(N)(0, 1)$ i.i.d.) satisfies $EE[Q] = sum_j lambda_j$ and $"Var"(Q) = 2 sum_j lambda_j^2$. Matching the first two moments of $Q$ to a scaled chi-square law $c chi_nu^2$ uniquely yields:
  $ nu = frac((sum_(j=1)^m lambda_j)^2, sum_(j=1)^m lambda_j^2) = frac(("Tr" bold(Sigma))^2, "Tr"(bold(Sigma)^2)) in [1, m], quad c = frac("Tr" bold(Sigma), nu). $
  Evaluating marginal mixture MDL with Satterthwaite degrees of freedom $nu$ and effective scale $hat(sigma)_("eff", K)^2 = frac("WCSS"_("clean")(K), N nu)$,
  $ cal(L)_("SID")(K) := -2 sum_(i=1)^N ln( sum_(k=1)^K hat(pi)_k (2 pi hat(sigma)_("eff", K)^2)^(-nu / 2) exp(-frac(norm(bold(z)_i - bold(mu)_k)_2^2, 2 hat(sigma)_("eff", K)^2)) ) + (K(nu + 1) - 1) ln N, $
  is strictly invariant under global scaling $bold(X) mapsto alpha bold(X) + bold(1) bold(c)^top$ ($alpha != 0$) and orthogonal transformations $bold(X) mapsto bold(X) bold(Q)$ ($bold(Q) in O(d)$), with zero tunable hyperparameters.
]

#proofbox[
  For $Y = c chi_nu^2$, $EE[Y] = c nu$ and $"Var"(Y) = 2 c^2 nu$, so $(EE[Y])^2 / "Var"(Y) = nu / 2$. Equating to $(EE[Q])^2 / "Var"(Q) = ("Tr" bold(Sigma))^2 / (2 "Tr"(bold(Sigma)^2))$ gives $nu$ and $c$ uniquely @satterthwaite1946approximate; Cauchy–Schwarz implies $1 <= nu <= m$. Under $bold(X) mapsto alpha bold(X) bold(Q) + bold(1) bold(c)^top$, centering removes $bold(c)$, orthogonal $bold(Q)$ preserves singular values $s_j$ and pairwise Euclidean distances identically, and scalar $alpha$ scales $s_j mapsto |alpha| s_j$, leaving $tau_("GD") / s_j$ and $nu$ invariant while shifting $cal(L)_("SID")(K)$ by an additive constant $N nu ln(alpha^2)$ independent of $K$.
]

= Empirical Evaluation on NVIDIA GB10 (Grace Blackwell)

#figure(
  image("../figures/fig1_highd_duality_collapse.png", width: 96%),
  caption: [Empirical verification of Theorem 2 across $d = 6 .. 1536$ (15 seeds/point, $N=480$). *(A)* Under isotropic ambient noise ($K^*=6$), Ambient `X-Means` collapses toward $1$ and Caliński–Harabasz collapses to $2.0$, whereas `SID-KMeans` maintains $hat(K) = 6.00$ at $d = 1536$. *(B)* On low-rank manifolds ($K^*=4$), Ambient `X-Means` inflates to $K_("max") = 12.0$ while `SID-KMeans` locks onto $4.00$ ($100%$ accuracy).]
)

#figure(
  image("../figures/fig2_null_overfitting_prevention.png", width: 92%),
  caption: [Unimodal null overfitting benchmark ($K^* = 1$, $N = 500$, 20 seeds/topology). Naive $"WCSS"$-BIC overfits $100%$ of 2D/anisotropic runs (confirming Theorem 1); Silhouette/CH/DB overfit $100%$ unconditionally; Ambient `X-Means` overfits $100%$ on 128D ellipsoids (`Mean K = 6.0`), whereas `SID-KMeans` achieves $0.0%$ overfitting across all three Euclidean topologies.]
)

#figure(
  image("../figures/fig3_transformer_embeddings_gb10.png", width: 98%),
  caption: [Real pretrained Transformer document clustering on NVIDIA GB10 across 60 `20 Newsgroups` corpora ($K^* in {1, 2, 4, 6, 8, 10}$, 5 random topic-subset corpora per $K^*$ for both `BGE-Base-768D` and `BGE-Large-1024D`). `SID-KMeans` tracks the ideal $hat(K) = K^*$ diagonal across both models while Gap Statistic over-splits curved unit-sphere manifolds and Caliński–Harabasz flatlines at $hat(K) = 2$.]
)

#figure(
  table(
    columns: (1.5fr, 0.95fr, 1.0fr, 0.85fr, 0.95fr, 1.0fr, 0.85fr),
    inset: 5.5pt,
    align: (left, right, right, right, right, right, right),
    stroke: 0.5pt + rgb("#cbd5e1"),
    fill: (_, row) => if row <= 1 { rgb("#eff6ff") } else { none },
    [*Method*], [*BGE-768D MAE* $arrow.b$], [*768D* $|hat(K)-K^*|<=1$], [*768D NMI* $arrow.t$], [*BGE-1024D MAE* $arrow.b$], [*1024D* $|hat(K)-K^*|<=1$], [*1024D NMI* $arrow.t$],
    [*SID-KMeans (Ours)*], [*0.20*], [*100.0%*], [*0.745*], [*0.23*], [*100.0%*], [*0.759*],
    [Ambient X-Means @pelleg2000xmeans], [0.70], [93.3%], [0.707], [0.53], [96.7%], [0.721],
    [Silhouette @rousseeuw1987silhouettes], [0.97], [73.3%], [0.715], [0.80], [86.7%], [0.752],
    [Caliński–Harabasz @calinski1974dendrite], [3.50], [33.3%], [0.435], [3.50], [30.0%], [0.449],
    [Davies–Bouldin], [4.03], [46.7%], [0.667], [4.33], [40.0%], [0.673],
    [Gap Statistic @tibshirani2001estimating], [4.77], [0.0%], [0.622], [5.13], [0.0%], [0.628],
  ),
  caption: [Aggregate performance across 60 real `20 Newsgroups` Transformer corpora ($K^* in {1, 2, 4, 6, 8, 10}$) on NVIDIA GB10. `SID-KMeans` achieves the lowest MAE and highest NMI on both 768D and 1024D models while running $4.6 times$ (768D) to $7.0 times$ (1024D) faster than the Gap Statistic.]
)

*1. Real 768D & 1024D Transformer Embeddings (60 Corpora on NVIDIA GB10).* Figure 4 and Table 1 present our flagship benchmark across `BAAI/bge-base-en-v1.5` ($768"D"$) and `BAAI/bge-large-en-v1.5` ($1024"D"$). `SID-KMeans` achieves *`MAE` $= 0.20$* on 768D ($2.3 times–3.5 times$ lower error than `X-Means`) and *`MAE` $= 0.23$* on 1024D, with *`100.0%` within-$plus.minus 1$ accuracy across all 60 corpora* and exact $hat(K) = 1.00$ on all 10 single-topic corpora. Notably, the widely recommended Gap Statistic fails completely on unit-sphere embeddings (`MAE` $= 4.77–5.13$, `0%` within-1 accuracy) because uniform hypercube reference draws in $RR^(1024)$ overestimate hollow-sphere dispersion, while Caliński–Harabasz flatlines at $hat(K) approx 2.0$ across all $K^* <= 10$ just as predicted by Theorem 2.

*2. Classical Benchmarks, Boundary Regimes, & Limitations.* On UCI `Wine` ($d=13, K^*=3$), `SID-KMeans` recovers $hat(K) = 3$ ($nu = 2.45$) whereas `X-Means` over-splits to $hat(K) = 7$; on `Iris` ($d=4, K^*=3$), `SID-KMeans` avoids `X-Means`'s extreme collinearity over-splitting ($hat(K) = 11$) and selects the $hat(K) = 2$ linearly separated macro-clusters ($nu = 1.71$). Two honest boundary conditions merit attention: *(i)* on synthetic wide-angle unit-sphere projections ($d=768$ cone in Figure 3), strong nonlinear spherical curvature induces a $hat(K) = 2$ hemisphere split under Euclidean SVD (suggesting tangent-space log-maps for wide geodesic cones); and *(ii)* in ultra-starved $N lt.double d$ pixel regimes (`Olivetti Faces`, $N = 50$ images in $d = 4096$ raw pixels), median singular-value thresholding acts conservatively ($hat(K) = 1$, vs. `X-Means`/Gap exploding to $hat(K) = 8–12$), underscoring that reliable unsupervised model selection requires $N ≳ K^* nu$.

= Conclusion

By replacing heuristic $"WCSS"$ penalties with Marchenko–Pastur bulk denoising, Satterthwaite quadratic-form degrees of freedom, and exact marginal mixture MDL, `SID-KMeans` resolves both unimodal Voronoi overfitting and high-dimensional embedding collapse without tunable hyperparameters. Reference implementations in Python and C\#/.NET are available at `https://github.com/AndrewRober/IntrinsicDimKMeans`.

#bibliography("references.bib", title: "References", style: "ieee")
