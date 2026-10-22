[← 30 Responsible AI: Ethics, Bias, Safety, and Governance](ch30-responsible-ai.md) · [Contents](../README.md#contents) · [Appendix B: Tools, Environment Setup, and Cheat Sheets →](ch32-appendix-b-tools-cheatsheets.md)

---

# Appendix A: Mathematical Notation and Reference

This appendix collects, in one place, the mathematical notation, formulas, and standard results that recur throughout *Ultimate AI/ML*. It is written for lookup rather than for a single linear read. When a chapter uses a symbol or a formula and you want the precise definition, the general rule, or a quick sanity check, this is where to find it. Nothing here is new material: every result appears in service of something concrete elsewhere in the book, and the relevant chapters are named where they matter.

A word on philosophy. Notation is a compression scheme for ideas. The value of a consistent notation is that once you have internalized it, a dense line of algebra reads like a sentence. The cost of an inconsistent notation is that every equation becomes a small decryption puzzle. This book commits to one set of conventions, stated in the first section below, and holds to them. When you see a bold lowercase letter you should think “vector” without pausing; when you see a capital letter in a matrix context you should think “matrix.” The rest of this appendix assumes those conventions.

## 30.14 A.1 Notation Conventions

### 30.14.1 A.1.1 Objects and Typography

The book distinguishes the *kind* of a mathematical object by its typeface and case. This distinction is load-bearing: it lets you read an expression like $`\mathbf{y} = \mathbf{X\beta} + \mathbf{\varepsilon}`$ and know immediately that a matrix is acting on a vector to produce a vector.

| Object | Convention | Examples |
|----|----|----|
| Scalar | Lowercase italic | $`a,\mspace{6mu} b,\mspace{6mu} x,\mspace{6mu}\lambda,\mspace{6mu}\eta`$ |
| Vector | Lowercase bold (column vector by default) | $`\mathbf{x},\mspace{6mu}\mathbf{w},\mspace{6mu}\mathbf{\theta}`$ |
| Matrix | Uppercase bold | $`\mathbf{A},\mspace{6mu}\mathbf{X},\mspace{6mu}\mathbf{W},\mspace{6mu}\mathbf{\Sigma}`$ |
| Tensor (order $`\geq 3`$) | Uppercase bold sans/script | $`\mathcal{X},\mspace{6mu}\mathcal{W}`$ |
| Set | Uppercase calligraphic | $`\mathcal{D},\mspace{6mu}\mathcal{S},\mspace{6mu}\mathcal{H}`$ |
| Random variable | Uppercase italic | $`X,\mspace{6mu} Y,\mspace{6mu} Z`$ |
| Realized value of a random variable | Lowercase italic | $`x,\mspace{6mu} y,\mspace{6mu} z`$ |
| Function | Italic, with arguments | $`f(x),\mspace{6mu} g(\mathbf{x}),\mspace{6mu}\sigma(z)`$ |
| Estimate / prediction | Hat accent | $`\widehat{y},\mspace{6mu}\widehat{\mathbf{\theta}},\mspace{6mu}\widehat{p}`$ |
| Optimal value | Star superscript | \$\boldsymbol{\theta}^{\\},\\ \mathbf{w}^{\\},\\ V^{\\}\$ |

By default a vector $`\mathbf{x} \in \mathbb{R}^{n}`$ is a **column vector**, an $`n \times 1`$ array. Its transpose $`\mathbf{x}^{\top}`$ is a row vector. This convention matters for matrix products: with $`\mathbf{X} \in \mathbb{R}^{m \times n}`$ holding $`m`$ examples in rows and $`n`$ features in columns (the standard data-matrix layout used from the Machine Learning Workflow chapter onward), a linear model produces predictions $`\widehat{\mathbf{y}} = \mathbf{Xw}`$ with $`\mathbf{w} \in \mathbb{R}^{n}`$.

### 30.14.2 A.1.2 Indexing

Indexing conventions differ between mathematics (1-based) and code (0-based in Python). The book uses 1-based indexing in formulas and notes where the corresponding code differs.

| Notation | Meaning |
|----|----|
| $`x_{i}`$ | The $`i`$-th component of vector $`\mathbf{x}`$ |
| $`A_{ij}`$ | Entry in row $`i`$, column $`j`$ of matrix $`\mathbf{A}`$ |
| $`\mathbf{a}_{i}`$ or $`\mathbf{A}_{i,:}`$ | The $`i`$-th row of $`\mathbf{A}`$ (as a vector) |
| $`\mathbf{A}_{:,j}`$ | The $`j`$-th column of $`\mathbf{A}`$ |
| $`\mathbf{x}^{(i)}`$ | The $`i`$-th training example (parenthesized superscript indexes the dataset) |
| $`y^{(i)}`$ | The label of the $`i`$-th training example |
| $`x_{j}^{(i)}`$ | The $`j`$-th feature of the $`i`$-th example |
| $`\theta_{t}`$ or $`\mathbf{\theta}^{\lbrack t\rbrack}`$ | The parameter value at optimization step $`t`$ |
| $`\mathbf{h}^{(l)}`$ | Activations at layer $`l`$ of a network (bracketed/parenthesized superscript for layer) |

The parenthesized superscript $`\mathbf{x}^{(i)}`$ is reserved for indexing *examples* so it is never confused with exponentiation. When an exponent is genuinely meant, it appears without parentheses: $`x^{2}`$ is $`x`$ squared, $`x^{(2)}`$ is the second example.

### 30.14.3 A.1.3 Number Sets and Spaces

| Symbol | Meaning |
|----|----|
| $`\mathbb{N}`$ | Natural numbers $`\{ 1,2,3,\ldots\}`$ |
| $`\mathbb{Z}`$ | Integers |
| $`\mathbb{Q}`$ | Rational numbers |
| $`\mathbb{R}`$ | Real numbers |
| $`\mathbb{R}_{\geq 0}`$ | Non-negative reals |
| $`\mathbb{R}^{n}`$ | $`n`$-dimensional real vector space (column vectors) |
| $`\mathbb{R}^{m \times n}`$ | Real matrices with $`m`$ rows and $`n`$ columns |
| $`\mathbb{C}`$ | Complex numbers |
| $`\lbrack n\rbrack`$ | The index set $`\{ 1,2,\ldots,n\}`$ |
| $`\{ 0,1\}`$ | Binary label set |
| $`\Delta^{k - 1}`$ | The probability simplex over $`k`$ classes (vectors of $`k`$ non-negative entries summing to 1) |

### 30.14.4 A.1.4 Set Notation

| Symbol | Meaning |
|----|----|
| $`\in ,\mspace{6mu} \notin`$ | Element of / not an element of |
| $`\subseteq ,\mspace{6mu} \subset`$ | Subset / proper subset |
| $`\cup ,\mspace{6mu} \cap`$ | Union / intersection |
| $`\backslash`$ | Set difference |
| $`\varnothing`$ | Empty set |
| $`|\mathcal{S}|`$ | Cardinality (number of elements) of $`\mathcal{S}`$ |
| $`\mathcal{A} \times \mathcal{B}`$ | Cartesian product |
| $`\{ x:P(x)\}`$ | Set-builder: all $`x`$ satisfying property $`P`$ |
| $`\mathcal{P}(\mathcal{S})`$ or $`2^{\mathcal{S}}`$ | Power set of $`\mathcal{S}`$ |
| $`\mathcal{D} = \{(\mathbf{x}^{(i)},y^{(i)})\}_{i = 1}^{m}`$ | A labeled training set of $`m`$ examples |

### 30.14.5 A.1.5 Machine-Learning-Specific Notation

Beyond the generic mathematical objects, the book uses a stable set of symbols for the recurring quantities of a learning problem. Fixing these once removes ambiguity when a chapter writes, say, $`J(\mathbf{\theta})`$ or $`\widehat{y}`$ without re-defining it.

| Symbol | Meaning |
|----|----|
| $`\mathbf{x},\mspace{6mu}\mathbf{x}^{(i)}`$ | Feature (input) vector; the $`i`$-th example’s features |
| $`y,\mspace{6mu} y^{(i)}`$ | Target (label); the $`i`$-th example’s label |
| $`\widehat{y},\mspace{6mu}\widehat{\mathbf{y}}`$ | Model prediction |
| $`\mathbf{X} \in \mathbb{R}^{m \times n}`$ | Design/data matrix: $`m`$ examples (rows), $`n`$ features (columns) |
| $`m`$ | Number of training examples |
| $`n`$ or $`d`$ | Number of features / input dimension |
| $`k`$ or $`K`$ | Number of classes, clusters, or components |
| $`\mathbf{\theta},\mspace{6mu}\mathbf{w},\mspace{6mu} b`$ | Model parameters; weight vector; bias (intercept) |
| $`\mathbf{W}^{(l)},\mspace{6mu}\mathbf{b}^{(l)}`$ | Weight matrix and bias of layer $`l`$ |
| $`f_{\mathbf{\theta}}(\mathbf{x})`$ or $`h(\mathbf{x})`$ | The hypothesis / model function |
| $`L,\mspace{6mu}\ell`$ | Loss (total / per-example) |
| $`J(\mathbf{\theta})`$ | Objective / cost function (mean loss plus regularization) |
| $`\eta,\mspace{6mu}\alpha`$ | Learning rate / step size |
| $`\lambda`$ | Regularization strength |
| $`\sigma( \cdot )`$ | Sigmoid or, in context, standard deviation |
| $`\pi,\mspace{6mu}\mathbf{\pi}`$ | Mixing weight / class probability (also policy in RL) |
| $`\gamma`$ | Discount factor (Reinforcement Learning chapters) |

Where a symbol is overloaded — $`\sigma`$ for sigmoid versus standard deviation, $`\pi`$ for a probability versus a policy, $`\lambda`$ for a regularizer versus an eigenvalue — the surrounding context disambiguates, and the book flags the reading in the relevant chapter.

### 30.14.6 A.1.6 Common Operators and Symbols

| Symbol | Meaning | Example |
|----|----|----|
| $`\sum_{i = 1}^{n}a_{i}`$ | Summation | $`\sum_{i = 1}^{3}i = 6`$ |
| $`\prod_{i = 1}^{n}a_{i}`$ | Product | $`\prod_{i = 1}^{3}i = 6`$ |
| $`\int_{a}^{b}f(x)\, dx`$ | Definite integral | area under $`f`$ from $`a`$ to $`b`$ |
| $`\lim_{x \rightarrow a}`$ | Limit | $`\lim_{x \rightarrow 0}\frac{\sin x}{x} = 1`$ |
| $`\frac{d}{dx},\mspace{6mu}\frac{\partial}{\partial x}`$ | Ordinary / partial derivative | slope of $`f`$ |
| $`\nabla f`$ | Gradient of $`f`$ (vector of partials) | $`\nabla f \in \mathbb{R}^{n}`$ |
| $`\nabla^{2}f`$ or $`\mathbf{H}`$ | Hessian (matrix of second partials) | $`\mathbf{H} \in \mathbb{R}^{n \times n}`$ |
| $`\mathbb{E}\lbrack \cdot \rbrack`$ | Expectation | $`\mathbb{E}\lbrack X\rbrack = \mu`$ |
| $`Var( \cdot )`$ | Variance | $`Var(X) = \sigma^{2}`$ |
| $`Cov( \cdot , \cdot )`$ | Covariance | $`Cov(X,Y)`$ |
| $`\mathbb{P}( \cdot ),\mspace{6mu} p( \cdot )`$ | Probability / density | $`p(x)`$ |
| $`\propto`$ | Proportional to | $`p(\theta \mid x) \propto p(x \mid \theta)\, p(\theta)`$ |
| $`\sim`$ | Distributed as | $`X \sim \mathcal{N}(0,1)`$ |
| $`\approx`$ | Approximately equal | $`\pi \approx 3.14`$ |
| $`\triangleq ,\mspace{6mu}: =`$ | Defined as | $`\bar{x}: = \frac{1}{n}\sum x_{i}`$ |
| $`\mathbb{1}\lbrack \cdot \rbrack`$ | Indicator (1 if condition true, else 0) | $`\mathbb{1}\lbrack y = \widehat{y}\rbrack`$ |
| $`\arg\max_{x},\mspace{6mu}\arg\min_{x}`$ | Argument that maximizes / minimizes | $`\widehat{y} = arg\max_{k}p_{k}`$ |
| $`\odot`$ | Elementwise (Hadamard) product | $`\mathbf{a} \odot \mathbf{b}`$ |
| $`\otimes`$ | Outer / Kronecker product | $`\mathbf{a}\mathbf{b}^{\top}`$ |
| $`\left\| \cdot \right\|`$ | Norm | $`\left\| \mathbf{x} \right\|_{2}`$ |
| $`\langle \cdot , \cdot \rangle`$ | Inner product | $`\langle\mathbf{a},\mathbf{b}\rangle`$ |
| $`\lfloor x\rfloor,\mspace{6mu}\lceil x\rceil`$ | Floor / ceiling | $`\lfloor 2.7\rfloor = 2`$ |
| $`\mathcal{O}( \cdot )`$ | Big-O asymptotic bound | $`\mathcal{O}(n^{2})`$ |

The gradient symbol $`\nabla`$ deserves a note because it appears on nearly every page of the training chapters. Read $`\nabla_{\mathbf{\theta}}L`$ as “the gradient of the loss $`L`$ with respect to the parameters $`\mathbf{\theta}`$” — a vector pointing in the direction of steepest increase of $`L`$, whose negation is the descent direction used in gradient descent.

A practical reading tip for the equations throughout the book: parse an expression by shape before meaning. Confirm the *kind* of each object from its typeface (Section A.1.1), then check that the shapes are compatible — a matrix-vector product needs matching inner dimensions, an elementwise operation needs identical shapes, and a scalar loss must ultimately reduce a vector or matrix to a single number. Nine times out of ten, a formula that looks intimidating is just a familiar operation applied to objects whose shapes you can verify at a glance.

## 30.15 A.2 Linear Algebra Reference

Linear algebra is the language of data, as established in the Mathematical Foundations chapter and used continuously through the deep learning chapters. This section is a condensed operational reference.

### 30.15.1 A.2.1 Core Operations

| Operation | Definition | Result shape |
|----|----|----|
| Vector addition | $`(\mathbf{a} + \mathbf{b})_{i} = a_{i} + b_{i}`$ | vector |
| Scalar multiplication | $`(c\mathbf{a})_{i} = c\, a_{i}`$ | vector |
| Dot / inner product | $`\mathbf{a}^{\top}\mathbf{b} = \sum_{i = 1}^{n}a_{i}b_{i}`$ | scalar |
| Outer product | $`(\mathbf{a}\mathbf{b}^{\top})_{ij} = a_{i}b_{j}`$ | $`m \times n`$ matrix |
| Hadamard product | $`(\mathbf{a} \odot \mathbf{b})_{i} = a_{i}b_{i}`$ | vector |
| Matrix-vector product | $`(\mathbf{Ax})_{i} = \sum_{j}^{}A_{ij}x_{j}`$ | vector |
| Matrix-matrix product | $`(\mathbf{AB})_{ij} = \sum_{k}^{}A_{ik}B_{kj}`$ | $`m \times p`$ matrix |

For the matrix product $`\mathbf{AB}`$ to be defined, the inner dimensions must agree: $`\mathbf{A} \in \mathbb{R}^{m \times n}`$ and $`\mathbf{B} \in \mathbb{R}^{n \times p}`$ give $`\mathbf{AB} \in \mathbb{R}^{m \times p}`$. Matrix multiplication is associative and distributive but **not commutative**: in general $`\mathbf{AB} \neq \mathbf{BA}`$.

### 30.15.2 A.2.2 The Dot Product and Geometry

The dot product connects algebra to geometry. For $`\mathbf{a},\mathbf{b} \in \mathbb{R}^{n}`$,

``` math
\mathbf{a}^{\top}\mathbf{b} = \left\| \mathbf{a} \right\|_{2}\,\left\| \mathbf{b} \right\|_{2}\cos\theta,
```

where $`\theta`$ is the angle between the vectors. Two nonzero vectors are **orthogonal** when $`\mathbf{a}^{\top}\mathbf{b} = 0`$. The **cosine similarity** used throughout the retrieval and embeddings discussions (the Retrieval-Augmented Generation chapter especially) is exactly

``` math
\text{cossim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}^{\top}\mathbf{b}}{\left\| \mathbf{a} \right\|_{2}\,\left\| \mathbf{b} \right\|_{2}} = cos\theta \in \lbrack - 1,1\rbrack.
```

### 30.15.3 A.2.3 Norms

A norm measures the “size” of a vector. The $`p`$-norm is $`\left\| \mathbf{x} \right\|_{p} = \left( \sum_{i}^{}|x_{i}|^{p} \right)^{1/p}`$.

| Norm | Formula | Used for |
|----|----|----|
| $`L_{1}`$ (Manhattan) | $`\left\| \mathbf{x} \right\|_{1} = \sum_{i}^{}|x_{i}|`$ | Lasso regularization (sparsity) |
| $`L_{2}`$ (Euclidean) | $`\left\| \mathbf{x} \right\|_{2} = \sqrt{\sum_{i}^{}x_{i}^{2}}`$ | Ridge, distances, most defaults |
| $`L_{\infty}`$ (max) | $`\left\| \mathbf{x} \right\|_{\infty} = \max_{i}|x_{i}|`$ | Robustness bounds |
| $`L_{0}`$ (“norm”) | number of nonzero entries | Counting sparsity (not a true norm) |
| Frobenius (matrix) | $`\left\| \mathbf{A} \right\|_{F} = \sqrt{\sum_{i,j}^{}A_{ij}^{2}}`$ | Matrix size, weight decay |

The squared $`L_{2}`$ norm $`\left\| \mathbf{x} \right\|_{2}^{2} = \mathbf{x}^{\top}\mathbf{x}`$ is preferred in optimization because it is differentiable everywhere and avoids the square root. The $`L_{1}`$ versus $`L_{2}`$ contrast underlies the regularization discussion in the Regression chapter: $`L_{1}`$ drives coefficients exactly to zero, $`L_{2}`$ shrinks them smoothly.

### 30.15.4 A.2.4 Special Matrices

| Matrix | Property |
|----|----|
| Identity $`\mathbf{I}`$ | $`\mathbf{IA} = \mathbf{AI} = \mathbf{A}`$; $`I_{ij} = \mathbb{1}\lbrack i = j\rbrack`$ |
| Diagonal $`\mathbf{D}`$ | Nonzero only on diagonal |
| Symmetric | $`\mathbf{A} = \mathbf{A}^{\top}`$ |
| Orthogonal $`\mathbf{Q}`$ | $`\mathbf{Q}^{\top}\mathbf{Q} = \mathbf{Q}\mathbf{Q}^{\top} = \mathbf{I}`$, so $`\mathbf{Q}^{- 1} = \mathbf{Q}^{\top}`$ |
| Positive definite (PD) | $`\mathbf{x}^{\top}\mathbf{Ax} > 0`$ for all $`\mathbf{x} \neq \mathbf{0}`$ |
| Positive semidefinite (PSD) | $`\mathbf{x}^{\top}\mathbf{Ax} \geq 0`$ for all $`\mathbf{x}`$ |

Covariance matrices (Section A.4) and Gram matrices $`\mathbf{X}^{\top}\mathbf{X}`$ are always PSD, a fact used repeatedly in the Dimensionality Reduction chapter.

### 30.15.5 A.2.5 Transpose, Trace, Determinant, Rank

| Quantity | Definition | Key properties |
|----|----|----|
| Transpose $`\mathbf{A}^{\top}`$ | $`(\mathbf{A}^{\top})_{ij} = A_{ji}`$ | $`(\mathbf{AB})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}`$ |
| Trace $`tr(\mathbf{A})`$ | $`\sum_{i}^{}A_{ii}`$ (sum of diagonal) | $`tr(\mathbf{AB}) = tr(\mathbf{BA})`$; equals $`\sum_{i}^{}\lambda_{i}`$ |
| Determinant $`det(\mathbf{A})`$ | signed volume scaling of $`\mathbf{A}`$ | $`det(\mathbf{AB}) = det(\mathbf{A})det(\mathbf{B})`$; equals $`\prod_{i}^{}\lambda_{i}`$ |
| Rank $`rank(\mathbf{A})`$ | number of linearly independent columns | $`\leq min(m,n)`$ |

A square matrix is **invertible** (nonsingular) if and only if $`det(\mathbf{A}) \neq 0`$, equivalently $`rank(\mathbf{A}) = n`$, equivalently all eigenvalues are nonzero.

### 30.15.6 A.2.6 Matrix Inverse

The inverse $`\mathbf{A}^{- 1}`$ satisfies $`\mathbf{A}\mathbf{A}^{- 1} = \mathbf{A}^{- 1}\mathbf{A} = \mathbf{I}`$. Useful identities:

``` math
(\mathbf{AB})^{- 1} = \mathbf{B}^{- 1}\mathbf{A}^{- 1},\quad\quad(\mathbf{A}^{\top})^{- 1} = (\mathbf{A}^{- 1})^{\top},\quad\quad(\mathbf{A}^{- 1})^{- 1} = \mathbf{A}.
```

For a $`2 \times 2`$ matrix,

``` math
\mathbf{A} = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix},\quad\quad\mathbf{A}^{- 1} = \frac{1}{ad - bc}\begin{pmatrix}
d & - b \\
 - c & a
\end{pmatrix}.
```

In practice — as the Regression chapter stresses — you rarely form an inverse explicitly. To solve $`\mathbf{Ax} = \mathbf{b}`$ you use a factorization-based solver (`np.linalg.solve`), which is faster and numerically more stable than computing $`\mathbf{A}^{- 1}\mathbf{b}`$.

### 30.15.7 A.2.7 The Pseudo-Inverse and Projections

When $`\mathbf{A}`$ is not square or not invertible, the **Moore–Penrose pseudo-inverse** $`\mathbf{A}^{+}`$ generalizes the inverse. For a tall full-column-rank matrix, $`\mathbf{A}^{+} = (\mathbf{A}^{\top}\mathbf{A})^{- 1}\mathbf{A}^{\top}`$, and the least-squares solution to the overdetermined system $`\mathbf{Ax} \approx \mathbf{b}`$ is $`\widehat{\mathbf{x}} = \mathbf{A}^{+}\mathbf{b}`$. Via the SVD $`\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}`$, the pseudo-inverse is $`\mathbf{A}^{+} = \mathbf{V}\mathbf{\Sigma}^{+}\mathbf{U}^{\top}`$, where $`\mathbf{\Sigma}^{+}`$ reciprocates the nonzero singular values — the numerically preferred route.

The **orthogonal projection** of $`\mathbf{b}`$ onto the column space of $`\mathbf{A}`$ is $`\mathbf{Pb}`$ with projection matrix $`\mathbf{P} = \mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{- 1}\mathbf{A}^{\top}`$. Projection matrices are symmetric and idempotent ($`\mathbf{P}^{2} = \mathbf{P}`$), and the “hat matrix” of linear regression that maps observed targets to fitted values is exactly this $`\mathbf{P}`$.

### 30.15.8 A.2.8 Eigenvalues and Eigenvectors

A nonzero vector $`\mathbf{v}`$ is an **eigenvector** of square $`\mathbf{A}`$ with **eigenvalue** $`\lambda`$ when

``` math
\mathbf{Av} = \lambda\mathbf{v}.
```

Geometrically, $`\mathbf{A}`$ merely stretches $`\mathbf{v}`$ by $`\lambda`$ without rotating it. For a symmetric matrix, the **spectral theorem** guarantees a full set of real eigenvalues and orthonormal eigenvectors, giving the eigendecomposition

``` math
\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\top},
```

where $`\mathbf{Q}`$ has the eigenvectors as columns and $`\mathbf{\Lambda} = diag(\lambda_{1},\ldots,\lambda_{n})`$. This is the machinery behind Principal Component Analysis in the Dimensionality Reduction chapter: the principal components are the eigenvectors of the covariance matrix, ordered by decreasing eigenvalue (variance explained).

### 30.15.9 A.2.9 Singular Value Decomposition (SVD)

Every matrix $`\mathbf{A} \in \mathbb{R}^{m \times n}`$ factorizes as

``` math
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top},
```

where $`\mathbf{U} \in \mathbb{R}^{m \times m}`$ and $`\mathbf{V} \in \mathbb{R}^{n \times n}`$ are orthogonal, and $`\mathbf{\Sigma} \in \mathbb{R}^{m \times n}`$ is diagonal with non-negative **singular values** $`\sigma_{1} \geq \sigma_{2} \geq \ldots \geq 0`$. The columns of $`\mathbf{U}`$ are left singular vectors, the columns of $`\mathbf{V}`$ right singular vectors. The SVD is the most numerically robust decomposition and underlies PCA, low-rank approximation, pseudo-inverses, and latent-factor recommender models. The best rank-$`k`$ approximation of $`\mathbf{A}`$ (Eckart–Young theorem) keeps the top $`k`$ singular values:

``` math
\mathbf{A}_{k} = \sum_{i = 1}^{k}\sigma_{i}\mathbf{u}_{i}\mathbf{v}_{i}^{\top}.
```

### 30.15.10 A.2.10 Key Identities Cheat Sheet

| Identity | Statement |
|----|----|
| Transpose of product | $`(\mathbf{AB})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}`$ |
| Inverse of product | $`(\mathbf{AB})^{- 1} = \mathbf{B}^{- 1}\mathbf{A}^{- 1}`$ |
| Trace cyclic | $`tr(\mathbf{ABC}) = tr(\mathbf{CAB})`$ |
| Trace of scalar | $`\mathbf{x}^{\top}\mathbf{Ax} = tr(\mathbf{Ax}\mathbf{x}^{\top})`$ |
| Determinant / trace via eigenvalues | $`det(\mathbf{A}) = \prod_{i}^{}\lambda_{i},\quad tr(\mathbf{A}) = \sum_{i}^{}\lambda_{i}`$ |
| Woodbury (matrix inversion lemma) | $`(\mathbf{A} + \mathbf{UCV})^{- 1} = \mathbf{A}^{- 1} - \mathbf{A}^{- 1}\mathbf{U}(\mathbf{C}^{- 1} + \mathbf{V}\mathbf{A}^{- 1}\mathbf{U})^{- 1}\mathbf{V}\mathbf{A}^{- 1}`$ |

## 30.16 A.3 Calculus Reference

Calculus supplies the slopes that drive learning. Every gradient-based optimizer in the Training Deep Networks chapter rests on the rules below.

### 30.16.1 A.3.1 Single-Variable Derivative Rules

| Rule | Statement |
|----|----|
| Constant | $`\frac{d}{dx}\, c = 0`$ |
| Power | $`\frac{d}{dx}\, x^{n} = nx^{n - 1}`$ |
| Exponential | $`\frac{d}{dx}\, e^{x} = e^{x}`$ |
| General exponential | $`\frac{d}{dx}\, a^{x} = a^{x}\ln a`$ |
| Natural log | $`\frac{d}{dx}\,\ln x = \frac{1}{x}`$ |
| Sum | $`(f + g)' = f' + g'`$ |
| Product | $`(fg)' = f'g + fg'`$ |
| Quotient | $`\left( \frac{f}{g} \right)' = \frac{f'g - fg'}{g^{2}}`$ |
| Chain | $`\frac{d}{dx}\, f(g(x)) = f'(g(x))\, g'(x)`$ |
| Sigmoid | $`\frac{d}{dx}\,\sigma(x) = \sigma(x)\,(1 - \sigma(x))`$ |
| Tanh | $`\frac{d}{dx}\, tanh(x) = 1 - \tanh^{2}(x)`$ |

### 30.16.2 A.3.2 Partial Derivatives, Gradient, Jacobian, Hessian

For a scalar function $`f:\mathbb{R}^{n} \rightarrow \mathbb{R}`$, the **partial derivative** $`\frac{\partial f}{\partial x_{i}}`$ treats all variables except $`x_{i}`$ as constants. Collecting them gives the **gradient**, a column vector:

``` math
\nabla f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_{1}},\ldots,\frac{\partial f}{\partial x_{n}} \right)^{\top} \in \mathbb{R}^{n}.
```

For a vector-valued function $`\mathbf{f}:\mathbb{R}^{n} \rightarrow \mathbb{R}^{m}`$, the **Jacobian** collects all first partials:

``` math
\mathbf{J}(\mathbf{x}) = \frac{\partial\mathbf{f}}{\partial\mathbf{x}},\quad\quad J_{ij} = \frac{\partial f_{i}}{\partial x_{j}} \in \mathbb{R}^{m \times n}.
```

The **Hessian** collects second partials of a scalar function and is symmetric for twice-continuously-differentiable $`f`$:

``` math
\mathbf{H}(\mathbf{x}) = \nabla^{2}f(\mathbf{x}),\quad\quad H_{ij} = \frac{\partial^{2}f}{\partial x_{i}\partial x_{j}} \in \mathbb{R}^{n \times n}.
```

The Hessian’s definiteness classifies critical points: PD means a local minimum, negative definite a local maximum, indefinite a saddle point. Saddle points, not local minima, are the dominant obstacle in high-dimensional deep learning, as discussed in the Training Deep Networks chapter.

### 30.16.3 A.3.3 The Multivariable Chain Rule and Backpropagation

Backpropagation, introduced in the Neural Network Fundamentals chapter, is the multivariable chain rule applied systematically. If $`z = f(\mathbf{y})`$ and $`\mathbf{y} = \mathbf{g}(\mathbf{x})`$, then

``` math
\frac{\partial z}{\partial x_{j}} = \sum_{i}^{}\frac{\partial z}{\partial y_{i}}\,\frac{\partial y_{i}}{\partial x_{j}},\quad\quad\text{or in vector form}\quad\quad\nabla_{\mathbf{x}}z = \mathbf{J}_{\mathbf{g}}^{\top}\,\nabla_{\mathbf{y}}z.
```

The transpose of the Jacobian propagates gradients backward through each layer — this is precisely the “backward pass.”

### 30.16.4 A.3.4 Matrix Calculus Identities

These identities recur when deriving gradients for linear models, least squares, and network layers. Here $`\mathbf{a}`$ and $`\mathbf{b}`$ are constant vectors, $`\mathbf{A}`$ a constant matrix, and the gradient is taken with respect to $`\mathbf{x}`$ (denominator-layout / gradient convention).

| Expression $`f(\mathbf{x})`$ | Gradient $`\nabla_{\mathbf{x}}f`$ |
|----|----|
| $`\mathbf{a}^{\top}\mathbf{x}`$ | $`\mathbf{a}`$ |
| $`\mathbf{x}^{\top}\mathbf{a}`$ | $`\mathbf{a}`$ |
| $`\mathbf{x}^{\top}\mathbf{x}`$ | $`2\mathbf{x}`$ |
| $`\mathbf{x}^{\top}\mathbf{Ax}`$ | $`(\mathbf{A} + \mathbf{A}^{\top})\mathbf{x}`$; $`= 2\mathbf{Ax}`$ if $`\mathbf{A}`$ symmetric |
| $`\mathbf{a}^{\top}\mathbf{Ax}`$ | $`\mathbf{A}^{\top}\mathbf{a}`$ |
| $`\left\| \mathbf{x} - \mathbf{a} \right\|_{2}^{2}`$ | $`2(\mathbf{x} - \mathbf{a})`$ |
| $`\left\| \mathbf{Ax} - \mathbf{b} \right\|_{2}^{2}`$ | $`2\mathbf{A}^{\top}(\mathbf{Ax} - \mathbf{b})`$ |

Matrix-valued derivatives (with respect to a matrix $`\mathbf{X}`$):

| Expression | Derivative w.r.t. $`\mathbf{X}`$ |
|----|----|
| $`tr(\mathbf{AX})`$ | $`\mathbf{A}^{\top}`$ |
| $`tr(\mathbf{X}^{\top}\mathbf{AX})`$ | $`(\mathbf{A} + \mathbf{A}^{\top})\mathbf{X}`$ |
| $`lndet(\mathbf{X})`$ | $`(\mathbf{X}^{- 1})^{\top}`$ |

The identity $`\nabla_{\mathbf{w}}\left\| \mathbf{Xw} - \mathbf{y} \right\|_{2}^{2} = 2\mathbf{X}^{\top}(\mathbf{Xw} - \mathbf{y})`$ is the workhorse of linear regression: setting it to zero yields the normal equations $`\mathbf{X}^{\top}\mathbf{Xw} = \mathbf{X}^{\top}\mathbf{y}`$.

### 30.16.5 A.3.5 Integrals Worth Knowing

Integration appears less often than differentiation in day-to-day machine learning, but a handful of results recur — normalizing constants of distributions, expectations, and the areas under evaluation curves.

| Integral | Result |
|----|----|
| $`\int x^{n}\, dx`$ | $`\frac{x^{n + 1}}{n + 1} + C`$ ($`n \neq - 1`$) |
| $`\int\frac{1}{x}\, dx`$ | $`ln|x| + C`$ |
| $`\int e^{x}\, dx`$ | $`e^{x} + C`$ |
| $`\int_{- \infty}^{\infty}e^{- x^{2}/2}\, dx`$ | $`\sqrt{2\pi}`$ (Gaussian normalizer) |
| $`\int_{0}^{\infty}x^{\alpha - 1}e^{- x}\, dx`$ | $`\Gamma(\alpha)`$ (Gamma function) |

Integration by parts, $`\int u\, dv = uv - \int v\, du`$, is the standard tool for deriving expectations and the moments in the distribution table of Section A.4.5.

### 30.16.6 A.3.6 Taylor Expansion

The Taylor expansion approximates a function near a point $`\mathbf{x}_{0}`$. In one dimension:

``` math
f(x) \approx f(x_{0}) + f'(x_{0})(x - x_{0}) + \frac{1}{2}f''(x_{0})(x - x_{0})^{2} + \ldots
```

In the multivariate case, to second order:

``` math
f(\mathbf{x}) \approx f(\mathbf{x}_{0}) + \nabla f(\mathbf{x}_{0})^{\top}(\mathbf{x} - \mathbf{x}_{0}) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_{0})^{\top}\mathbf{H}(\mathbf{x}_{0})(\mathbf{x} - \mathbf{x}_{0}).
```

The first-order term justifies gradient descent (move opposite the gradient); the second-order term justifies Newton and quasi-Newton methods, which use curvature from the Hessian.

## 30.17 A.4 Probability Reference

Probability is the language of uncertainty, developed in the Mathematical Foundations chapter and central to every probabilistic model in the book.

### 30.17.1 A.4.1 Axioms (Kolmogorov)

For a sample space $`\Omega`$ and events $`A \subseteq \Omega`$:

1.  **Non-negativity:** $`\mathbb{P}(A) \geq 0`$.
2.  **Normalization:** $`\mathbb{P}(\Omega) = 1`$.
3.  **Additivity:** for disjoint events, $`\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B)`$.

From these: $`\mathbb{P}(A^{c}) = 1 - \mathbb{P}(A)`$ and the general union rule $`\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)`$.

### 30.17.2 A.4.2 Conditional Probability, Independence, Bayes

``` math
\mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)},\quad\quad\mathbb{P}(B) > 0.
```

Events are **independent** when $`\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)`$, equivalently $`\mathbb{P}(A \mid B) = \mathbb{P}(A)`$. The **chain rule** generalizes: $`\mathbb{P}(A_{1},\ldots,A_{n}) = \prod_{i}^{}\mathbb{P}(A_{i} \mid A_{1},\ldots,A_{i - 1})`$. The **law of total probability**, for a partition $`\{ B_{j}\}`$, is $`\mathbb{P}(A) = \sum_{j}^{}\mathbb{P}(A \mid B_{j})\mathbb{P}(B_{j})`$.

**Bayes’ theorem** — the engine of Naive Bayes classifiers and all Bayesian inference — inverts a conditional:

``` math
\underset{\text{posterior}}{\underbrace{\mathbb{P}(H \mid D)}} = \frac{\overset{\text{likelihood}}{\overbrace{\mathbb{P}(D \mid H)}}\mspace{6mu}\overset{\text{prior}}{\overbrace{\mathbb{P}(H)}}}{\underset{\text{evidence}}{\underbrace{\mathbb{P}(D)}}},\quad\quad\mathbb{P}(D) = \sum_{H'}^{}\mathbb{P}(D \mid H')\mathbb{P}(H').
```

### 30.17.3 A.4.3 Expectation, Variance, Covariance

| Quantity | Discrete | Continuous |
|----|----|----|
| Expectation | $`\mathbb{E}\lbrack X\rbrack = \sum_{x}^{}x\, p(x)`$ | $`\int x\, p(x)\, dx`$ |
| Variance | $`Var(X) = \mathbb{E}\lbrack(X - \mu)^{2}\rbrack = \mathbb{E}\lbrack X^{2}\rbrack - \mu^{2}`$ | same |
| Covariance | $`Cov(X,Y) = \mathbb{E}\lbrack(X - \mu_{X})(Y - \mu_{Y})\rbrack`$ | same |
| Correlation | $`\rho = \frac{Cov(X,Y)}{\sigma_{X}\sigma_{Y}} \in \lbrack - 1,1\rbrack`$ | same |

Key rules (with constants $`a,b`$):

``` math
\mathbb{E}\lbrack aX + b\rbrack = a\,\mathbb{E}\lbrack X\rbrack + b,\quad\quad\mathbb{E}\lbrack X + Y\rbrack = \mathbb{E}\lbrack X\rbrack + \mathbb{E}\lbrack Y\rbrack\ \text{(always)},
```

``` math
Var(aX + b) = a^{2}\, Var(X),\quad\quad Var(X + Y) = Var(X) + Var(Y) + 2\, Cov(X,Y).
```

Linearity of expectation holds even for dependent variables; the variance-of-a-sum rule simplifies to $`Var(X) + Var(Y)`$ only when $`X`$ and $`Y`$ are uncorrelated. For independent $`X,Y`$: $`\mathbb{E}\lbrack XY\rbrack = \mathbb{E}\lbrack X\rbrack\mathbb{E}\lbrack Y\rbrack`$.

### 30.17.4 A.4.4 Limit Theorems

Two theorems underwrite nearly all of statistics and the reliability of sample-based estimates.

The **Law of Large Numbers (LLN)** states that the sample mean of iid draws converges to the true mean as the sample grows: $`{\bar{X}}_{n} \rightarrow \mu`$ as $`n \rightarrow \infty`$. It is why more data yields more trustworthy averages and why Monte Carlo estimation works.

The **Central Limit Theorem (CLT)** states that the standardized sample mean converges to a standard Gaussian regardless of the underlying distribution’s shape (given finite variance):

``` math
\frac{{\bar{X}}_{n} - \mu}{\sigma/\sqrt{n}}\overset{d}{\rightarrow}\mathcal{N}(0,1).
```

The CLT is the reason confidence intervals and many hypothesis tests can assume approximate normality of a sample mean, and it explains why Gaussian noise is such a defensible default modeling assumption.

### 30.17.5 A.4.5 Common Distributions

The following table is a reference for the distributions used across the book — Bernoulli and Categorical for classification targets, Gaussian for noise and priors, Poisson for counts, Beta and Gamma as conjugate priors.

| Distribution | Support | PMF / PDF | Mean | Variance |
|----|----|----|----|----|
| Bernoulli$`(p)`$ | $`x \in \{ 0,1\}`$ | $`p^{x}(1 - p)^{1 - x}`$ | $`p`$ | $`p(1 - p)`$ |
| Binomial$`(n,p)`$ | $`x \in \{ 0,\ldots,n\}`$ | $`\left( \frac{n}{x} \right)p^{x}(1 - p)^{n - x}`$ | $`np`$ | $`np(1 - p)`$ |
| Categorical$`(\mathbf{\pi})`$ | $`x \in \{ 1,\ldots,k\}`$ | $`\prod_{j = 1}^{k}\pi_{j}^{\mathbb{1}\lbrack x = j\rbrack}`$ | — | — |
| Poisson$`(\lambda)`$ | $`x \in \{ 0,1,2,\ldots\}`$ | $`\frac{\lambda^{x}e^{- \lambda}}{x!}`$ | $`\lambda`$ | $`\lambda`$ |
| Uniform$`(a,b)`$ | $`x \in \lbrack a,b\rbrack`$ | $`\frac{1}{b - a}`$ | $`\frac{a + b}{2}`$ | $`\frac{(b - a)^{2}}{12}`$ |
| Gaussian$`(\mu,\sigma^{2})`$ | $`x \in \mathbb{R}`$ | $`\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left( - \frac{(x - \mu)^{2}}{2\sigma^{2}} \right)`$ | $`\mu`$ | $`\sigma^{2}`$ |
| Exponential$`(\lambda)`$ | $`x \geq 0`$ | $`\lambda e^{- \lambda x}`$ | $`\frac{1}{\lambda}`$ | $`\frac{1}{\lambda^{2}}`$ |
| Beta$`(\alpha,\beta)`$ | $`x \in \lbrack 0,1\rbrack`$ | $`\frac{x^{\alpha - 1}(1 - x)^{\beta - 1}}{B(\alpha,\beta)}`$ | $`\frac{\alpha}{\alpha + \beta}`$ | $`\frac{\alpha\beta}{(\alpha + \beta)^{2}(\alpha + \beta + 1)}`$ |
| Gamma$`(\alpha,\beta)`$ | $`x \geq 0`$ | $`\frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha - 1}e^{- \beta x}`$ | $`\frac{\alpha}{\beta}`$ | $`\frac{\alpha}{\beta^{2}}`$ |

Here $`B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}`$ is the Beta function and $`\Gamma( \cdot )`$ the Gamma function ($`\Gamma(n) = (n - 1)!`$ for positive integers). The Categorical distribution generalizes Bernoulli to $`k`$ outcomes with probability vector $`\mathbf{\pi} \in \Delta^{k - 1}`$; its per-class mean is the vector $`\mathbf{\pi}`$ itself under one-hot encoding. Beta is the conjugate prior for the Bernoulli/Binomial parameter $`p`$; Gamma is conjugate for the Poisson rate and the Gaussian precision.

### 30.17.6 A.4.6 The Multivariate Gaussian

For $`\mathbf{x} \in \mathbb{R}^{d}`$ with mean $`\mathbf{\mu} \in \mathbb{R}^{d}`$ and covariance $`\mathbf{\Sigma} \in \mathbb{R}^{d \times d}`$ (symmetric PD):

``` math
\mathcal{N}(\mathbf{x} \mid \mathbf{\mu},\mathbf{\Sigma}) = \frac{1}{(2\pi)^{d/2}\,|\mathbf{\Sigma}|^{1/2}}\exp\left( - \frac{1}{2}(\mathbf{x} - \mathbf{\mu})^{\top}\mathbf{\Sigma}^{- 1}(\mathbf{x} - \mathbf{\mu}) \right).
```

The quadratic form $`(\mathbf{x} - \mathbf{\mu})^{\top}\mathbf{\Sigma}^{- 1}(\mathbf{x} - \mathbf{\mu})`$ is the squared **Mahalanobis distance**, a covariance-aware distance that appears in Gaussian mixture models (Clustering chapter) and anomaly detection. Linear transformations preserve Gaussianity: if $`\mathbf{x} \sim \mathcal{N}(\mathbf{\mu},\mathbf{\Sigma})`$ then $`\mathbf{Ax} + \mathbf{b} \sim \mathcal{N}(\mathbf{A\mu} + \mathbf{b},\,\mathbf{A}\mathbf{\Sigma}\mathbf{A}^{\top})`$.

### 30.17.7 A.4.7 Distances and Similarity Measures

Distances quantify how far apart two points or distributions are, and they recur in clustering (Clustering chapter), nearest-neighbor methods, and retrieval (Retrieval-Augmented Generation chapter). This consolidated table collects the ones the book uses.

| Measure | Formula | Notes |
|----|----|----|
| Euclidean ($`L_{2}`$) | $`\sqrt{\sum_{i}^{}(a_{i} - b_{i})^{2}}`$ | Default geometric distance |
| Manhattan ($`L_{1}`$) | $`\sum_{i}^{}|a_{i} - b_{i}|`$ | Grid/robust distance |
| Minkowski ($`p`$) | $`\left( \sum_{i}^{}|a_{i} - b_{i}|^{p} \right)^{1/p}`$ | Generalizes $`L_{1}`$, $`L_{2}`$, $`L_{\infty}`$ |
| Chebyshev ($`L_{\infty}`$) | $`\max_{i}|a_{i} - b_{i}|`$ | Largest single-coordinate gap |
| Cosine distance | $`1 - \frac{\mathbf{a}^{\top}\mathbf{b}}{\left\| \mathbf{a} \right\|\,\left\| \mathbf{b} \right\|}`$ | Ignores magnitude; embeddings |
| Mahalanobis | $`\sqrt{(\mathbf{a} - \mathbf{b})^{\top}\mathbf{\Sigma}^{- 1}(\mathbf{a} - \mathbf{b})}`$ | Covariance-aware |
| Hamming | $`\sum_{i}^{}\mathbb{1}\lbrack a_{i} \neq b_{i}\rbrack`$ | Categorical / binary strings |
| Jaccard | $`1 - \frac{|A \cap B|}{|A \cup B|}`$ | Set overlap |

A valid **metric** must satisfy non-negativity, identity of indiscernibles, symmetry, and the triangle inequality $`d(a,c) \leq d(a,b) + d(b,c)`$. Cosine distance and KL divergence violate one or more of these (the triangle inequality and symmetry, respectively) and so are dissimilarity measures rather than true metrics — a distinction that matters when an algorithm’s guarantees assume a metric.

## 30.18 A.5 Statistics Reference

Statistics turns data into trustworthy conclusions and parameter estimates, as developed in the Mathematical Foundations and Supervised Learning Evaluation chapters.

### 30.18.1 A.5.1 Estimators and Their Quality

An **estimator** $`\widehat{\theta}`$ is a function of the data that approximates a population parameter $`\theta`$. Its quality is decomposed by the **bias-variance decomposition** of mean squared error:

``` math
\mathbb{E}\left\lbrack (\widehat{\theta} - \theta)^{2} \right\rbrack = \underset{\text{bias}^{2}}{\underbrace{(\mathbb{E}\lbrack\widehat{\theta}\rbrack - \theta)^{2}}} + \underset{\text{variance}}{\underbrace{Var(\widehat{\theta})}}.
```

An estimator is **unbiased** if $`\mathbb{E}\lbrack\widehat{\theta}\rbrack = \theta`$ and **consistent** if $`\widehat{\theta} \rightarrow \theta`$ as the sample size grows. This same decomposition, applied to predictions, drives the bias-variance discussion of model complexity in the Supervised Learning Evaluation chapter. The sample mean $`\bar{x} = \frac{1}{n}\sum x_{i}`$ is unbiased for $`\mu`$; the sample variance uses the Bessel correction $`s^{2} = \frac{1}{n - 1}\sum(x_{i} - \bar{x})^{2}`$ to remain unbiased for $`\sigma^{2}`$.

### 30.18.2 A.5.2 Maximum Likelihood Estimation (MLE)

MLE chooses parameters that make the observed data most probable. Given iid data $`\mathcal{D} = \{ x^{(i)}\}`$ and a model $`p(x \mid \mathbf{\theta})`$:

``` math
{\widehat{\mathbf{\theta}}}_{\text{MLE}} = arg\max_{\mathbf{\theta}}\prod_{i = 1}^{m}p(x^{(i)} \mid \mathbf{\theta}) = arg\max_{\mathbf{\theta}}\sum_{i = 1}^{m}\log p(x^{(i)} \mid \mathbf{\theta}).
```

The **recipe**: (1) write the likelihood; (2) take the log to convert the product into a sum; (3) differentiate the log-likelihood with respect to $`\mathbf{\theta}`$; (4) set to zero and solve (or optimize numerically). Taking the log is not cosmetic: it turns a fragile product of many small probabilities (which underflows to zero numerically) into a stable sum, and it does not change the location of the maximum because $`\log`$ is monotonic. MLE under a Gaussian noise model recovers least squares; under a Bernoulli/Categorical model it recovers cross-entropy. This unifying result — that the loss functions of Section A.8 are negative log-likelihoods of a chosen noise model — appears in both the Regression and Classification chapters and is one of the most useful organizing ideas in the book.

### 30.18.3 A.5.3 Maximum a Posteriori (MAP)

MAP adds a prior $`p(\mathbf{\theta})`$ and maximizes the posterior:

``` math
{\widehat{\mathbf{\theta}}}_{\text{MAP}} = arg\max_{\mathbf{\theta}}\left\lbrack \log p(\mathcal{D} \mid \mathbf{\theta}) + logp(\mathbf{\theta}) \right\rbrack.
```

The prior term acts as regularization: a Gaussian prior on weights yields $`L_{2}`$ (ridge) regularization, and a Laplace prior yields $`L_{1}`$ (lasso). MAP reduces to MLE as the prior becomes uninformative (flat).

### 30.18.4 A.5.4 Confidence Intervals

A $`(1 - \alpha)`$ confidence interval is a data-derived range that would contain the true parameter in a $`(1 - \alpha)`$ fraction of repeated experiments. For a mean with known $`\sigma`$:

``` math
\bar{x} \pm z_{1 - \alpha/2}\,\frac{\sigma}{\sqrt{n}},\quad\quad z_{0.975} \approx 1.96\text{for a 95\%interval}.
```

With unknown $`\sigma`$, replace $`\sigma`$ with the sample standard deviation $`s`$ and $`z`$ with the Student-$`t`$ critical value $`t_{1 - \alpha/2,\, n - 1}`$.

### 30.18.5 A.5.5 Common Hypothesis Tests

A hypothesis test compares a null hypothesis $`H_{0}`$ against an alternative $`H_{1}`$, computing a **p-value** (probability of a statistic at least as extreme as observed, under $`H_{0}`$) and rejecting $`H_{0}`$ when $`p < \alpha`$.

| Test | Use case | Statistic |
|----|----|----|
| One-sample $`t`$-test | Mean vs. a reference value | $`t = \frac{\bar{x} - \mu_{0}}{s/\sqrt{n}}`$ |
| Two-sample $`t`$-test | Compare two group means | pooled/Welch $`t`$ |
| Paired $`t`$-test | Before/after on same units | $`t`$ on differences |
| $`\chi^{2}`$ test | Categorical association / goodness of fit | $`\sum\frac{(O - E)^{2}}{E}`$ |
| ANOVA ($`F`$-test) | Compare $`\geq 3`$ group means | ratio of variances |
| Mann–Whitney U | Nonparametric two-group comparison | rank-based |
| Kolmogorov–Smirnov | Compare distributions | max CDF gap |

Two error types govern testing: a **Type I error** (false positive) rejects a true $`H_{0}`$ with probability $`\alpha`$; a **Type II error** (false negative) fails to reject a false $`H_{0}`$ with probability $`\beta`$, and **power** is $`1 - \beta`$. This vocabulary mirrors the classification error framework — false positives and false negatives — used to define precision and recall in the Supervised Learning Evaluation chapter; the two are the same idea viewed from statistics and from machine learning respectively.

A caution the book returns to repeatedly: a p-value is **not** the probability that $`H_{0}`$ is true, and statistical significance is not practical significance. With large datasets, negligibly small effects can be “significant.” Always pair a test with an effect size and, where possible, a confidence interval, and correct for multiple comparisons (for example with the Bonferroni adjustment $`\alpha' = \alpha/t`$ across $`t`$ tests) when running many tests at once.

## 30.19 A.6 Information Theory Reference

Information theory quantifies uncertainty and the “distance” between distributions. It supplies the loss functions for classification and the objectives for generative models (Generative Models chapter). All logarithms below use base 2 for bits or base $`e`$ for nats; machine learning conventionally uses natural log (nats).

| Quantity | Definition | Interpretation |
|----|----|----|
| Entropy | $`H(p) = - \sum_{x}^{}p(x)logp(x)`$ | Average uncertainty in $`p`$ |
| Cross-entropy | $`H(p,q) = - \sum_{x}^{}p(x)logq(x)`$ | Cost of coding $`p`$ using $`q`$ |
| KL divergence | $`D_{KL}(p\,\|\, q) = \sum_{x}^{}p(x)log\frac{p(x)}{q(x)}`$ | Extra cost of using $`q`$ for $`p`$ |
| Mutual information | $`I(X;Y) = \sum_{x,y}^{}p(x,y)log\frac{p(x,y)}{p(x)p(y)}`$ | Shared information between $`X`$ and $`Y`$ |

These are tied together by the identities

``` math
H(p,q) = H(p) + D_{KL}(p\,\|\, q),\quad\quad I(X;Y) = H(X) - H(X \mid Y) = D_{KL}\left( p(x,y)\,\|\, p(x)p(y) \right).
```

Two properties matter constantly. First, $`D_{KL}(p\,\|\, q) \geq 0`$ with equality iff $`p = q`$ (Gibbs’ inequality), so KL is a valid “closeness” measure — though not a true metric, being asymmetric. Second, minimizing cross-entropy $`H(p,q)`$ over $`q`$ is equivalent to minimizing $`D_{KL}(p\,\|\, q)`$, since $`H(p)`$ does not depend on $`q`$. This is exactly why the cross-entropy loss (Section A.8) is the natural training objective for classifiers: it minimizes the divergence between the true label distribution and the model’s predicted distribution. For continuous variables, replace sums with integrals to obtain differential entropy.

The asymmetry of KL is not a defect but a modeling choice with consequences. Minimizing the **forward** KL $`D_{KL}(p\,\|\, q)`$ (used in maximum-likelihood training) is “mean-seeking”: $`q`$ is penalized wherever $`p`$ has mass, so $`q`$ spreads to cover all modes of $`p`$. Minimizing the **reverse** KL $`D_{KL}(q\,\|\, p)`$ (used in variational inference, the Generative Models chapter) is “mode-seeking”: $`q`$ can safely ignore regions where $`p`$ has mass, so it tends to lock onto a single mode. Recognizing which direction an objective uses explains much of the qualitative behavior of generative and variational methods.

A symmetric alternative used when a genuine distance between distributions is wanted is the **Jensen–Shannon divergence**, $`\text{JSD}(p,q) = \frac{1}{2}D_{KL}(p\,\|\, m) + \frac{1}{2}D_{KL}(q\,\|\, m)`$ with $`m = \frac{1}{2}(p + q)`$, which is bounded and symmetric and appears in the original formulation of generative adversarial networks.

## 30.20 A.7 Optimization Reference

Optimization is how models learn. The Training Deep Networks chapter treats these ideas at length; here is the compact reference.

### 30.20.1 A.7.1 Gradient Descent

To minimize $`L(\mathbf{\theta})`$, iterate in the direction of steepest descent:

``` math
\mathbf{\theta}_{t + 1} = \mathbf{\theta}_{t} - \eta\,\nabla_{\mathbf{\theta}}L(\mathbf{\theta}_{t}),
```

where $`\eta > 0`$ is the **learning rate**. Variants differ in how the gradient is estimated and how the step is adapted:

| Variant | Update flavor |
|----|----|
| Batch GD | Gradient over the full dataset each step |
| Stochastic GD (SGD) | Gradient from one example (or minibatch) |
| Momentum | $`\mathbf{v}_{t + 1} = \gamma\mathbf{v}_{t} + \nabla L`$; step uses $`\mathbf{v}`$ |
| RMSProp | Scales step by a running average of squared gradients |
| Adam | Combines momentum and RMSProp with bias correction |
| Newton’s method | $`\mathbf{\theta}_{t + 1} = \mathbf{\theta}_{t} - \mathbf{H}^{- 1}\nabla L`$ (second-order) |

The learning rate is the single most consequential hyperparameter: too large and the iterates diverge or oscillate; too small and training crawls. For a convex quadratic with Hessian eigenvalues in $`\lbrack\ell,u\rbrack`$, gradient descent converges only when $`\eta < 2/u`$, and the convergence rate is governed by the **condition number** $`\kappa = u/\ell`$ — a fact that motivates input normalization and batch normalization in the Training Deep Networks chapter, both of which improve conditioning so that a single learning rate works well across all directions.

### 30.20.2 A.7.2 Convexity

A set is **convex** if the line segment between any two of its points stays inside it. A function $`f`$ is convex if

``` math
f\left( \lambda\mathbf{x} + (1 - \lambda)\mathbf{y} \right) \leq \lambda f(\mathbf{x}) + (1 - \lambda)f(\mathbf{y}),\quad\quad\lambda \in \lbrack 0,1\rbrack.
```

Equivalently, $`f`$ is convex iff its Hessian is PSD everywhere. The decisive property: for a convex function, **any local minimum is a global minimum**. Linear and logistic regression have convex losses (so optimization is well-behaved); neural networks do not, which is why initialization, learning-rate schedules, and stochasticity matter so much in the deep learning chapters.

### 30.20.3 A.7.3 Constrained Optimization: Lagrangian and KKT

To minimize $`f(\mathbf{x})`$ subject to equality constraints $`h_{i}(\mathbf{x}) = 0`$ and inequality constraints $`g_{j}(\mathbf{x}) \leq 0`$, form the **Lagrangian**:

``` math
\mathcal{L}(\mathbf{x},\mathbf{\lambda},\mathbf{\mu}) = f(\mathbf{x}) + \sum_{i}^{}\lambda_{i}h_{i}(\mathbf{x}) + \sum_{j}^{}\mu_{j}g_{j}(\mathbf{x}).
```

At an optimum, the **Karush–Kuhn–Tucker (KKT) conditions** hold:

| Condition | Statement |
|----|----|
| Stationarity | $`\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}`$ |
| Primal feasibility | $`h_{i}(\mathbf{x}) = 0,\quad g_{j}(\mathbf{x}) \leq 0`$ |
| Dual feasibility | $`\mu_{j} \geq 0`$ |
| Complementary slackness | $`\mu_{j}\, g_{j}(\mathbf{x}) = 0`$ |

These conditions are the theoretical backbone of Support Vector Machines (Classification with SVMs chapter): complementary slackness is exactly what makes only the support vectors (points on the margin) carry nonzero dual coefficients.

### 30.20.4 A.7.4 Regularization Penalties

Regularization adds a penalty on parameter size to the objective, trading a little bias for reduced variance (Section A.5.1). The general form is $`J(\mathbf{\theta}) = L(\mathbf{\theta}) + \lambda\, R(\mathbf{\theta})`$ with strength $`\lambda \geq 0`$.

| Penalty | $`R(\mathbf{\theta})`$ | Effect | Bayesian prior |
|----|----|----|----|
| $`L_{2}`$ (ridge / weight decay) | $`\left\| \mathbf{\theta} \right\|_{2}^{2}`$ | Shrinks weights smoothly | Gaussian |
| $`L_{1}`$ (lasso) | $`\left\| \mathbf{\theta} \right\|_{1}`$ | Drives weights to exactly zero (sparsity) | Laplace |
| Elastic net | $`\alpha\left\| \mathbf{\theta} \right\|_{1} + (1 - \alpha)\left\| \mathbf{\theta} \right\|_{2}^{2}`$ | Sparse but grouped | mixture |

As Section A.5.3 notes, these penalties are exactly the log-priors of a MAP estimate, which is why the Regression chapter can present ridge and lasso either as optimization penalties or as Bayesian priors and mean the same thing.

## 30.21 A.8 Common Loss Functions

Losses measure how wrong a prediction is; training minimizes their average over the data. Below, $`y`$ is the true value, $`\widehat{y}`$ the prediction; for classification $`p`$ is the predicted probability, and for multiclass $`\mathbf{p}`$ is the predicted distribution with true one-hot label $`\mathbf{y}`$.

| Loss | Formula | Gradient (w.r.t. prediction) | Used for |
|----|----|----|----|
| Squared error (MSE) | $`\frac{1}{2}(\widehat{y} - y)^{2}`$ | $`\widehat{y} - y`$ | Regression |
| Absolute error (MAE) | $`|\widehat{y} - y|`$ | $`sign(\widehat{y} - y)`$ | Robust regression |
| Huber$`(\delta)`$ | $`\left\{ \begin{matrix}
\frac{1}{2}(\widehat{y} - y)^{2} & |\widehat{y} - y| \leq \delta \\
\delta|\widehat{y} - y| - \frac{1}{2}\delta^{2} & \text{else}
\end{matrix} \right.\`$ | clamped residual $`clip(\widehat{y} - y, - \delta,\delta)`$ | Regression, outlier-tolerant |
| Binary cross-entropy | $`- \left\lbrack y\log p + (1 - y)log(1 - p) \right\rbrack`$ | $`\frac{p - y}{p(1 - p)}`$; $`= p - y`$ w.r.t. logit | Binary classification |
| Categorical cross-entropy | $`- \sum_{k}^{}y_{k}\log p_{k}`$ | $`p_{k} - y_{k}`$ w.r.t. logit $`z_{k}`$ | Multiclass classification |
| Hinge | $`max(0,\, 1 - y\,\widehat{y})`$, $`y \in \{ - 1, + 1\}`$ | $`- y`$ if $`y\widehat{y} < 1`$ else $`0`$ | SVM |
| KL divergence | $`\sum_{k}^{}y_{k}\log\frac{y_{k}}{p_{k}}`$ | $`- \frac{y_{k}}{p_{k}}`$ | Distribution matching, distillation |

The remarkable fact worth memorizing: when cross-entropy is composed with a softmax (multiclass) or sigmoid (binary) output, the gradient with respect to the pre-activation logit collapses to the clean form $`p - y`$. This cancellation is why these output-loss pairings are the universal default, and it is derived in the Neural Network Fundamentals chapter. The factor of $`\frac{1}{2}`$ in MSE and Huber is a convention that cancels the 2 from differentiation.

## 30.22 A.9 Common Activation Functions

Activations inject nonlinearity into networks. The table pairs each with its derivative, which the backward pass needs.

| Activation | Formula $`f(z)`$ | Derivative $`f'(z)`$ | Range |
|----|----|----|----|
| Sigmoid (logistic) | $`\sigma(z) = \frac{1}{1 + e^{- z}}`$ | $`\sigma(z)(1 - \sigma(z))`$ | $`(0,1)`$ |
| Tanh | $`tanh(z) = \frac{e^{z} - e^{- z}}{e^{z} + e^{- z}}`$ | $`1 - \tanh^{2}(z)`$ | $`( - 1,1)`$ |
| ReLU | $`max(0,z)`$ | $`\mathbb{1}\lbrack z > 0\rbrack`$ | $`\lbrack 0,\infty)`$ |
| Leaky ReLU$`(\alpha)`$ | $`max(\alpha z,z)`$ | $`1`$ if $`z > 0`$ else $`\alpha`$ | $`( - \infty,\infty)`$ |
| ELU$`(\alpha)`$ | $`z`$ if $`z > 0`$ else $`\alpha(e^{z} - 1)`$ | $`1`$ if $`z > 0`$ else $`\alpha e^{z}`$ | $`( - \alpha,\infty)`$ |
| GELU | $`z\,\Phi(z)`$ ($`\Phi`$ = standard normal CDF) | $`\Phi(z) + z\,\phi(z)`$ | $`\approx ( - 0.17,\infty)`$ |
| Softplus | $`ln(1 + e^{z})`$ | $`\sigma(z)`$ | $`(0,\infty)`$ |
| Softmax | $`\text{softmax}(\mathbf{z})_{i} = \frac{e^{z_{i}}}{\sum_{j}^{}e^{z_{j}}}`$ | $`\frac{\partial p_{i}}{\partial z_{j}} = p_{i}(\delta_{ij} - p_{j})`$ | $`\Delta^{k - 1}`$ |

ReLU is the default hidden-layer activation for its cheap gradient and resistance to vanishing gradients; sigmoid and softmax are output activations that produce probabilities; GELU is the standard choice inside transformers (Transformers chapter). The Kronecker delta $`\delta_{ij}`$ in the softmax Jacobian is 1 when $`i = j`$ and 0 otherwise.

A note on the two failure modes these functions can cause, both discussed in the Training Deep Networks chapter. Saturating activations (sigmoid, tanh) have derivatives that approach zero for large $`|z|`$; stacked across many layers, these near-zero factors multiply during backpropagation and produce the **vanishing gradient** problem. ReLU avoids saturation on the positive side but introduces the **dying ReLU** problem: a unit stuck at $`z < 0`$ has zero gradient forever and stops learning. Leaky ReLU, ELU, and GELU are engineered compromises that keep a nonzero gradient for negative inputs while retaining most of ReLU’s benefits.

## 30.23 A.10 Useful Identities

A grab-bag of algebraic relations that surface constantly in derivations.

### 30.23.1 A.10.1 Logarithm and Exponential Rules

| Identity | Statement |
|----|----|
| Product | $`log(ab) = loga + logb`$ |
| Quotient | $`log(a/b) = loga - logb`$ |
| Power | $`log(a^{b}) = b\log a`$ |
| Change of base | $`\log_{b}a = \frac{\ln a}{\ln b}`$ |
| Exp of sum | $`e^{a + b} = e^{a}e^{b}`$ |
| Log-of-exp | $`ln(e^{x}) = x,\quad e^{\ln x} = x`$ |
| Log-sum-exp (stable) | $`\log\sum_{i}^{}e^{z_{i}} = c + log\sum_{i}^{}e^{z_{i} - c},\ c = \max_{i}z_{i}`$ |

The log-sum-exp trick prevents overflow when computing softmax and cross-entropy and is used inside virtually every deep-learning framework’s numerically stable implementations.

### 30.23.2 A.10.2 Softmax and Sigmoid Relations

| Identity | Statement |
|----|----|
| Shift invariance | $`\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})`$ |
| Softmax sums to one | $`\sum_{i}^{}\text{softmax}(\mathbf{z})_{i} = 1`$ |
| Binary softmax = sigmoid | $`\text{softmax}(\lbrack z,0\rbrack)_{1} = \sigma(z)`$ |
| Sigmoid reflection | $`\sigma( - z) = 1 - \sigma(z)`$ |
| Logit inverse | $`\sigma^{- 1}(p) = ln\frac{p}{1 - p}`$ (the log-odds / logit) |
| Sigmoid derivative | $`\sigma'(z) = \sigma(z)\left( 1 - \sigma(z) \right)`$ |
| Tanh–sigmoid link | $`tanh(z) = 2\sigma(2z) - 1`$ |

The shift-invariance of softmax is the theoretical justification for subtracting the maximum logit before exponentiating: it changes nothing mathematically while preventing numerical overflow.

### 30.23.3 A.10.3 Combinatorics and Miscellany

| Identity | Statement |
|----|----|
| Binomial coefficient | $`\left( \frac{n}{k} \right) = \frac{n!}{k!(n - k)!}`$ |
| Geometric series | $`\sum_{k = 0}^{\infty}r^{k} = \frac{1}{1 - r}`$ for $`|r| < 1`$ |
| Sum of first $`n`$ integers | $`\sum_{k = 1}^{n}k = \frac{n(n + 1)}{2}`$ |
| Gaussian integral | $`\int_{- \infty}^{\infty}e^{- x^{2}}\, dx = \sqrt{\pi}`$ |
| Jensen’s inequality (convex $`f`$) | $`f(\mathbb{E}\lbrack X\rbrack) \leq \mathbb{E}\lbrack f(X)\rbrack`$ |

## 30.24 A.11 How to Use This Appendix

Treat this appendix as a companion you keep open while reading the rest of the book, not as a chapter to be studied front to back. Three habits make it most useful.

First, when a chapter introduces a symbol you are unsure of, come to **Section A.1** and confirm its kind and convention before reading further; a single misread symbol can derail a whole derivation. Second, when a chapter states a result without full derivation — the normal equations in the Regression chapter, the softmax-cross-entropy gradient in the Neural Network Fundamentals chapter, the PCA eigendecomposition in the Dimensionality Reduction chapter — trace it back to the matching identity here (Sections A.3, A.8, A.2 respectively) to see the general rule it instantiates. Third, when you implement something and the numbers look wrong, use the loss, activation, and identity tables (Sections A.8 through A.10) as a checklist: a misplaced sign, a forgotten $`\frac{1}{2}`$, or an unstable exponential is very often the culprit, and the correct form is one lookup away.

The notation and results here are deliberately the *lingua franca* of modern machine learning, not idiosyncratic to this book. Once these conventions are second nature, papers, documentation, and other references become far easier to read — which is, ultimately, the point of learning a language.

---

[← 30 Responsible AI: Ethics, Bias, Safety, and Governance](ch30-responsible-ai.md) · [Contents](../README.md#contents) · [Appendix B: Tools, Environment Setup, and Cheat Sheets →](ch32-appendix-b-tools-cheatsheets.md)
