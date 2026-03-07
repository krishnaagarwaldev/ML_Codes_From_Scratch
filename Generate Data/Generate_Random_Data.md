Good Krishna 👍

Now I’ll give you **ALL possible variations to generate synthetic binary classification data** (like your lab question).

We will focus only on **data generation part** (Step 2).

---

# ✅ 1️⃣ Gaussian Distribution (Most Common – Your Lab Type)

### Two separate clusters

<pre class="overflow-visible! px-0!" data-start="291" data-end="623"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼn">import</span><span></span><span class="ͼt">numpy</span><span></span><span class="ͼn">as</span><span></span><span class="ͼt">np</span><br/><br/><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>seed(</span><span class="ͼq">0</span><span>)</span><br/><br/><span class="ͼt">n</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">400</span><br/><span class="ͼt">half</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">n</span><span></span><span class="ͼn">//</span><span></span><span class="ͼq">2</span><br/><br/><span class="ͼl"># Class 0 centered at (-2, -2)</span><br/><span class="ͼt">class0</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼt">half</span><span>, </span><span class="ͼq">2</span><span>) </span><span class="ͼn">+</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>array([</span><span class="ͼn">-</span><span class="ͼq">2</span><span>, </span><span class="ͼn">-</span><span class="ͼq">2</span><span>])</span><br/><br/><span class="ͼl"># Class 1 centered at (2, 2)</span><br/><span class="ͼt">class1</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼt">half</span><span>, </span><span class="ͼq">2</span><span>) </span><span class="ͼn">+</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>array([</span><span class="ͼq">2</span><span>, </span><span class="ͼq">2</span><span>])</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">class0</span><span>, </span><span class="ͼt">class1</span><span>))</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">np</span><span class="ͼn">.</span><span>zeros((</span><span class="ͼt">half</span><span>,</span><span class="ͼq">1</span><span>)), </span><span class="ͼt">np</span><span class="ͼn">.</span><span>ones((</span><span class="ͼt">half</span><span>,</span><span class="ͼq">1</span><span>))))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Linearly separable

✔ Most used in exams

---

# ✅ 2️⃣ Gaussian with Different Covariance (Elliptical Clusters)

<pre class="overflow-visible! px-0!" data-start="743" data-end="1048"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">mean0</span><span></span><span class="ͼn">=</span><span> [</span><span class="ͼn">-</span><span class="ͼq">2</span><span>, </span><span class="ͼn">-</span><span class="ͼq">2</span><span>]</span><br/><span class="ͼt">cov0</span><span></span><span class="ͼn">=</span><span> [[</span><span class="ͼq">1</span><span>, </span><span class="ͼq">0.8</span><span>], [</span><span class="ͼq">0.8</span><span>, </span><span class="ͼq">1</span><span>]]</span><br/><br/><span class="ͼt">mean1</span><span></span><span class="ͼn">=</span><span> [</span><span class="ͼq">2</span><span>, </span><span class="ͼq">2</span><span>]</span><br/><span class="ͼt">cov1</span><span></span><span class="ͼn">=</span><span> [[</span><span class="ͼq">1</span><span>, </span><span class="ͼn">-</span><span class="ͼq">0.6</span><span>], [</span><span class="ͼn">-</span><span class="ͼq">0.6</span><span>, </span><span class="ͼq">1</span><span>]]</span><br/><br/><span class="ͼt">class0</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>multivariate_normal(</span><span class="ͼt">mean0</span><span>, </span><span class="ͼt">cov0</span><span>, </span><span class="ͼq">200</span><span>)</span><br/><span class="ͼt">class1</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>multivariate_normal(</span><span class="ͼt">mean1</span><span>, </span><span class="ͼt">cov1</span><span>, </span><span class="ͼq">200</span><span>)</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">class0</span><span>, </span><span class="ͼt">class1</span><span>))</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">np</span><span class="ͼn">.</span><span>zeros((</span><span class="ͼq">200</span><span>,</span><span class="ͼq">1</span><span>)), </span><span class="ͼt">np</span><span class="ͼn">.</span><span>ones((</span><span class="ͼq">200</span><span>,</span><span class="ͼq">1</span><span>))))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Rotated elliptical clusters

✔ More realistic data

---

# ✅ 3️⃣ Uniform Distribution Clusters

<pre class="overflow-visible! px-0!" data-start="1151" data-end="1336"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">class0</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>uniform(</span><span class="ͼn">-</span><span class="ͼq">4</span><span>, </span><span class="ͼn">-</span><span class="ͼq">1</span><span>, (</span><span class="ͼq">200</span><span>,</span><span class="ͼq">2</span><span>))</span><br/><span class="ͼt">class1</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>uniform(</span><span class="ͼq">1</span><span>, </span><span class="ͼq">4</span><span>, (</span><span class="ͼq">200</span><span>,</span><span class="ͼq">2</span><span>))</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">class0</span><span>, </span><span class="ͼt">class1</span><span>))</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">np</span><span class="ͼn">.</span><span>zeros((</span><span class="ͼq">200</span><span>,</span><span class="ͼq">1</span><span>)), </span><span class="ͼt">np</span><span class="ͼn">.</span><span>ones((</span><span class="ͼq">200</span><span>,</span><span class="ͼq">1</span><span>))))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Simple square clusters

---

# ✅ 4️⃣ Circular Non-Linear Data

<pre class="overflow-visible! px-0!" data-start="1405" data-end="1659"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">theta</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>linspace(</span><span class="ͼq">0</span><span>, </span><span class="ͼq">2</span><span class="ͼn">*</span><span class="ͼt">np</span><span class="ͼn">.</span><span>pi, </span><span class="ͼq">200</span><span>)</span><br/><br/><span class="ͼt">r1</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">1</span><br/><span class="ͼt">r2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">3</span><br/><br/><span class="ͼt">class0</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>c_[</span><span class="ͼt">r1</span><span class="ͼn">*</span><span class="ͼt">np</span><span class="ͼn">.</span><span>cos(</span><span class="ͼt">theta</span><span>), </span><span class="ͼt">r1</span><span class="ͼn">*</span><span class="ͼt">np</span><span class="ͼn">.</span><span>sin(</span><span class="ͼt">theta</span><span>)]</span><br/><span class="ͼt">class1</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>c_[</span><span class="ͼt">r2</span><span class="ͼn">*</span><span class="ͼt">np</span><span class="ͼn">.</span><span>cos(</span><span class="ͼt">theta</span><span>), </span><span class="ͼt">r2</span><span class="ͼn">*</span><span class="ͼt">np</span><span class="ͼn">.</span><span>sin(</span><span class="ͼt">theta</span><span>)]</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">class0</span><span>, </span><span class="ͼt">class1</span><span>))</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">np</span><span class="ͼn">.</span><span>zeros((</span><span class="ͼq">200</span><span>,</span><span class="ͼq">1</span><span>)), </span><span class="ͼt">np</span><span class="ͼn">.</span><span>ones((</span><span class="ͼq">200</span><span>,</span><span class="ͼq">1</span><span>))))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Not linearly separable

✔ Logistic regression will struggle

---

# ✅ 5️⃣ XOR Pattern (Non-Linear Hard Case)

<pre class="overflow-visible! px-0!" data-start="1776" data-end="1891"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">400</span><span>,</span><span class="ͼq">2</span><span>)</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>logical_xor(</span><span class="ͼt">X</span><span>[:,</span><span class="ͼq">0</span><span>] </span><span class="ͼn">></span><span></span><span class="ͼq">0</span><span>, </span><span class="ͼt">X</span><span>[:,</span><span class="ͼq">1</span><span>] </span><span class="ͼn">></span><span></span><span class="ͼq">0</span><span>)</span><span class="ͼn">.</span><span>astype(</span><span class="ͼt">int</span><span>)</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">y</span><span class="ͼn">.</span><span>reshape(</span><span class="ͼn">-</span><span class="ͼq">1</span><span>,</span><span class="ͼq">1</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Classic ML example

✔ Needs polynomial features

---

# ✅ 6️⃣ Imbalanced Dataset

<pre class="overflow-visible! px-0!" data-start="1980" data-end="2159"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">class0</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">350</span><span>,</span><span class="ͼq">2</span><span>) </span><span class="ͼn">+</span><span> [</span><span class="ͼn">-</span><span class="ͼq">2</span><span>,</span><span class="ͼn">-</span><span class="ͼq">2</span><span>]</span><br/><span class="ͼt">class1</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">50</span><span>,</span><span class="ͼq">2</span><span>) </span><span class="ͼn">+</span><span> [</span><span class="ͼq">2</span><span>,</span><span class="ͼq">2</span><span>]</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">class0</span><span>, </span><span class="ͼt">class1</span><span>))</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">np</span><span class="ͼn">.</span><span>zeros((</span><span class="ͼq">350</span><span>,</span><span class="ͼq">1</span><span>)), </span><span class="ͼt">np</span><span class="ͼn">.</span><span>ones((</span><span class="ͼq">50</span><span>,</span><span class="ͼq">1</span><span>))))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ 90% vs 10%

✔ Tests model robustness

---

# ✅ 7️⃣ Using Random Linear Boundary (General Method)

Instead of manually creating clusters:

<pre class="overflow-visible! px-0!" data-start="2304" data-end="2453"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>seed(</span><span class="ͼq">0</span><span>)</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">400</span><span>,</span><span class="ͼq">2</span><span>)</span><br/><br/><span class="ͼt">real_w</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>array([[</span><span class="ͼq">1</span><span>], [</span><span class="ͼn">-</span><span class="ͼq">2</span><span>]])</span><br/><span class="ͼt">bias</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">0.5</span><br/><br/><span class="ͼt">z</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">X</span><span></span><span class="ͼn">@</span><span></span><span class="ͼt">real_w</span><span></span><span class="ͼn">+</span><span></span><span class="ͼt">bias</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span> (</span><span class="ͼt">z</span><span></span><span class="ͼn">></span><span></span><span class="ͼq">0</span><span>)</span><span class="ͼn">.</span><span>astype(</span><span class="ͼt">int</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Automatically creates linear separable data

✔ Very general method

---

# ✅ 8️⃣ With Noise Added to Labels

<pre class="overflow-visible! px-0!" data-start="2569" data-end="2671"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">noise_idx</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>choice(</span><span class="ͼq">400</span><span>, </span><span class="ͼq">40</span><span>)  </span><span class="ͼl"># flip 10% labels</span><br/><span class="ͼt">y</span><span>[</span><span class="ͼt">noise_idx</span><span>] </span><span class="ͼn">=</span><span></span><span class="ͼq">1</span><span></span><span class="ͼn">-</span><span></span><span class="ͼt">y</span><span>[</span><span class="ͼt">noise_idx</span><span>]</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Simulates real-world errors

---

# ✅ 9️⃣ Spiral Dataset (Very Advanced)

<pre class="overflow-visible! px-0!" data-start="2751" data-end="3050"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">n</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">200</span><br/><span class="ͼt">theta</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>sqrt(</span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>rand(</span><span class="ͼt">n</span><span>))</span><span class="ͼn">*</span><span class="ͼq">2</span><span class="ͼn">*</span><span class="ͼt">np</span><span class="ͼn">.</span><span>pi</span><br/><br/><span class="ͼt">r_a</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">2</span><span class="ͼn">*</span><span class="ͼt">theta</span><span></span><span class="ͼn">+</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>pi</span><br/><span class="ͼt">data_a</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>c_[</span><span class="ͼt">np</span><span class="ͼn">.</span><span>cos(</span><span class="ͼt">theta</span><span>)</span><span class="ͼn">*</span><span class="ͼt">r_a</span><span>, </span><span class="ͼt">np</span><span class="ͼn">.</span><span>sin(</span><span class="ͼt">theta</span><span>)</span><span class="ͼn">*</span><span class="ͼt">r_a</span><span>]</span><br/><br/><span class="ͼt">r_b</span><span></span><span class="ͼn">=</span><span></span><span class="ͼn">-</span><span class="ͼq">2</span><span class="ͼn">*</span><span class="ͼt">theta</span><span></span><span class="ͼn">-</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>pi</span><br/><span class="ͼt">data_b</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>c_[</span><span class="ͼt">np</span><span class="ͼn">.</span><span>cos(</span><span class="ͼt">theta</span><span>)</span><span class="ͼn">*</span><span class="ͼt">r_b</span><span>, </span><span class="ͼt">np</span><span class="ͼn">.</span><span>sin(</span><span class="ͼt">theta</span><span>)</span><span class="ͼn">*</span><span class="ͼt">r_b</span><span>]</span><br/><br/><span class="ͼt">X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">data_a</span><span>, </span><span class="ͼt">data_b</span><span>))</span><br/><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>vstack((</span><span class="ͼt">np</span><span class="ͼn">.</span><span>zeros((</span><span class="ͼt">n</span><span>,</span><span class="ͼq">1</span><span>)), </span><span class="ͼt">np</span><span class="ͼn">.</span><span>ones((</span><span class="ͼt">n</span><span>,</span><span class="ͼq">1</span><span>))))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Highly non-linear

✔ Needs neural network

---

# 🔥 For Your IIITDM Lab Question

Your question specifically says:

> Gaussian distributions centered at (-2,-2) and (2,2)

So **Variation 1 is the correct expected answer.**

---

# 🎯 Important for Viva

They may ask:

1. Why Gaussian?

   → Because many real-world features follow normal distribution.
2. Why shuffle?

   → To remove ordering bias.
3. Why random seed?

   → For reproducibility.

---

If you want next, I can give:

* ✅ Proper clean final version of full logistic regression lab solution
* ✅ Visualization code for all variations
* ✅ Train-test splitting methods
* ✅ Why your previous file had mistake in labels

Tell me what you want next Krishna 👌

give all random numpy functions with all variations codes

Excellent Krishna 🔥

Now I’ll give you **ALL important NumPy random functions** with  **clear variations + short working code examples** .

This will help in:

* ML lab exams
* Synthetic dataset generation
* Interview questions
* Understanding distributions

---

# 🎯 1️⃣ Uniform Distribution

### 📌 `np.random.rand()` → Uniform [0,1)

<pre class="overflow-visible! px-0!" data-start="346" data-end="447"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼn">import</span><span></span><span class="ͼt">numpy</span><span></span><span class="ͼn">as</span><span></span><span class="ͼt">np</span><br/><br/><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>rand(</span><span class="ͼq">5</span><span>)        </span><span class="ͼl"># 1D</span><br/><span class="ͼt">x2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>rand(</span><span class="ͼq">3</span><span>,</span><span class="ͼq">2</span><span>)     </span><span class="ͼl"># 2D</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

### 📌 `np.random.uniform(low, high, size)`

<pre class="overflow-visible! px-0!" data-start="499" data-end="581"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>uniform(</span><span class="ͼq">5</span><span>, </span><span class="ͼq">10</span><span>, </span><span class="ͼq">5</span><span>)</span><br/><span class="ͼt">x2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>uniform(</span><span class="ͼn">-</span><span class="ͼq">2</span><span>, </span><span class="ͼq">2</span><span>, (</span><span class="ͼq">3</span><span>,</span><span class="ͼq">2</span><span>))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Equal probability everywhere

---

# 🎯 2️⃣ Normal (Gaussian) Distribution

### 📌 `np.random.randn()` → Mean=0, Std=1

<pre class="overflow-visible! px-0!" data-start="707" data-end="769"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">5</span><span>)</span><br/><span class="ͼt">x2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">3</span><span>,</span><span class="ͼq">2</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

### 📌 `np.random.normal(mean, std, size)`

<pre class="overflow-visible! px-0!" data-start="820" data-end="899"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>normal(</span><span class="ͼq">10</span><span>, </span><span class="ͼq">2</span><span>, </span><span class="ͼq">5</span><span>)</span><br/><span class="ͼt">x2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>normal(</span><span class="ͼq">0</span><span>, </span><span class="ͼq">1</span><span>, (</span><span class="ͼq">4</span><span>,</span><span class="ͼq">3</span><span>))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Most used in ML

---

# 🎯 3️⃣ Integer Random

### 📌 `np.random.randint(low, high, size)`

<pre class="overflow-visible! px-0!" data-start="997" data-end="1080"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randint(</span><span class="ͼq">0</span><span>, </span><span class="ͼq">10</span><span>, </span><span class="ͼq">5</span><span>)</span><br/><span class="ͼt">x2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randint(</span><span class="ͼq">1</span><span>, </span><span class="ͼq">100</span><span>, (</span><span class="ͼq">3</span><span>,</span><span class="ͼq">2</span><span>))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ For labels

✔ For classification

---

# 🎯 4️⃣ Random Choice

### 📌 `np.random.choice()`

<pre class="overflow-visible! px-0!" data-start="1179" data-end="1369"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼl"># From list</span><br/><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>choice([</span><span class="ͼq">0</span><span>,</span><span class="ͼq">1</span><span>], </span><span class="ͼq">10</span><span>)</span><br/><br/><span class="ͼl"># With probabilities</span><br/><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>choice([</span><span class="ͼq">0</span><span>,</span><span class="ͼq">1</span><span>], </span><span class="ͼq">10</span><span>, </span><span class="ͼt">p</span><span class="ͼn">=</span><span>[</span><span class="ͼq">0.8</span><span>,</span><span class="ͼq">0.2</span><span>])</span><br/><br/><span class="ͼl"># Without replacement</span><br/><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>choice(</span><span class="ͼq">10</span><span>, </span><span class="ͼq">5</span><span>, </span><span class="ͼt">replace</span><span class="ͼn">=</span><span class="ͼq">False</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ For imbalanced datasets

---

# 🎯 5️⃣ Shuffle

### 📌 `np.random.shuffle()`

<pre class="overflow-visible! px-0!" data-start="1453" data-end="1505"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>arange(</span><span class="ͼq">10</span><span>)</span><br/><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>shuffle(</span><span class="ͼt">x</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

⚠ Shuffles in-place

---

### 📌 `np.random.permutation()`

<pre class="overflow-visible! px-0!" data-start="1569" data-end="1612"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>permutation(</span><span class="ͼq">10</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Returns new shuffled array

---

# 🎯 6️⃣ Multivariate Normal (Advanced Gaussian)

<pre class="overflow-visible! px-0!" data-start="1701" data-end="1802"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">mean</span><span></span><span class="ͼn">=</span><span> [</span><span class="ͼq">0</span><span>,</span><span class="ͼq">0</span><span>]</span><br/><span class="ͼt">cov</span><span></span><span class="ͼn">=</span><span> [[</span><span class="ͼq">1</span><span>,</span><span class="ͼq">0.8</span><span>],[</span><span class="ͼq">0.8</span><span>,</span><span class="ͼq">1</span><span>]]</span><br/><br/><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>multivariate_normal(</span><span class="ͼt">mean</span><span>, </span><span class="ͼt">cov</span><span>, </span><span class="ͼq">200</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ For rotated clusters

---

# 🎯 7️⃣ Binomial Distribution

<pre class="overflow-visible! px-0!" data-start="1867" data-end="1924"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>binomial(</span><span class="ͼt">n</span><span class="ͼn">=</span><span class="ͼq">10</span><span>, </span><span class="ͼt">p</span><span class="ͼn">=</span><span class="ͼq">0.5</span><span>, </span><span class="ͼt">size</span><span class="ͼn">=</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Number of successes

---

# 🎯 8️⃣ Bernoulli Distribution

<pre class="overflow-visible! px-0!" data-start="1989" data-end="2037"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>binomial(</span><span class="ͼq">1</span><span>, </span><span class="ͼq">0.7</span><span>, </span><span class="ͼq">10</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ 0/1 labels

---

# 🎯 9️⃣ Exponential Distribution

<pre class="overflow-visible! px-0!" data-start="2095" data-end="2153"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>exponential(</span><span class="ͼt">scale</span><span class="ͼn">=</span><span class="ͼq">1.0</span><span>, </span><span class="ͼt">size</span><span class="ͼn">=</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Time between events

---

# 🎯 🔟 Poisson Distribution

<pre class="overflow-visible! px-0!" data-start="2215" data-end="2265"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>poisson(</span><span class="ͼt">lam</span><span class="ͼn">=</span><span class="ͼq">3</span><span>, </span><span class="ͼt">size</span><span class="ͼn">=</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Count data

---

# 🎯 1️⃣1️⃣ Beta Distribution

<pre class="overflow-visible! px-0!" data-start="2319" data-end="2369"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>beta(</span><span class="ͼt">a</span><span class="ͼn">=</span><span class="ͼq">2</span><span>, </span><span class="ͼt">b</span><span class="ͼn">=</span><span class="ͼq">5</span><span>, </span><span class="ͼt">size</span><span class="ͼn">=</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Probability modeling

---

# 🎯 1️⃣2️⃣ Gamma Distribution

<pre class="overflow-visible! px-0!" data-start="2434" data-end="2493"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>gamma(</span><span class="ͼt">shape</span><span class="ͼn">=</span><span class="ͼq">2</span><span>, </span><span class="ͼt">scale</span><span class="ͼn">=</span><span class="ͼq">2</span><span>, </span><span class="ͼt">size</span><span class="ͼn">=</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 🎯 1️⃣3️⃣ Lognormal Distribution

<pre class="overflow-visible! px-0!" data-start="2536" data-end="2598"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>lognormal(</span><span class="ͼt">mean</span><span class="ͼn">=</span><span class="ͼq">0</span><span>, </span><span class="ͼt">sigma</span><span class="ͼn">=</span><span class="ͼq">1</span><span>, </span><span class="ͼt">size</span><span class="ͼn">=</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 🎯 1️⃣4️⃣ Random Bytes

<pre class="overflow-visible! px-0!" data-start="2631" data-end="2668"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>bytes(</span><span class="ͼq">10</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 🎯 1️⃣5️⃣ Set Random Seed

<pre class="overflow-visible! px-0!" data-start="2704" data-end="2736"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>seed(</span><span class="ͼq">42</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Reproducibility

---

# 🔥 Modern Recommended Method (NumPy Generator API)

Instead of old `np.random.*`, use:

<pre class="overflow-visible! px-0!" data-start="2854" data-end="2977"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">rng</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>default_rng(</span><span class="ͼq">42</span><span>)</span><br/><br/><span class="ͼt">x</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">rng</span><span class="ͼn">.</span><span>normal(</span><span class="ͼq">0</span><span>,</span><span class="ͼq">1</span><span>,(</span><span class="ͼq">5</span><span>,</span><span class="ͼq">2</span><span>))</span><br/><span class="ͼt">x2</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">rng</span><span class="ͼn">.</span><span>uniform(</span><span class="ͼq">0</span><span>,</span><span class="ͼq">10</span><span>,</span><span class="ͼq">5</span><span>)</span><br/><span class="ͼt">x3</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">rng</span><span class="ͼn">.</span><span>integers(</span><span class="ͼq">0</span><span>,</span><span class="ͼq">10</span><span>,</span><span class="ͼq">5</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

✔ Better practice (NumPy ≥1.17)

---

# 🧠 Important for ML Lab

Most commonly used:

* rand
* randn
* normal
* uniform
* randint
* choice
* multivariate_normal
* shuffle
* permutation

---

# 🎯 If You Want Next

I can give:

* 🔥 Full comparison table of distributions
* 🔥 Visualization of each distribution
* 🔥 When to use which in ML
* 🔥 Logistic regression dataset generation combinations

Tell me what you want Krishna 👌

[ ]
