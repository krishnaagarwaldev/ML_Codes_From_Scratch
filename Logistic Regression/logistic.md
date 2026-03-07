Krishna 👌 since you’re preparing ML from scratch, I’ll explain your Logistic Regression code  **step-by-step** , point out  **mistakes** , and show you the  **correct and improved version** .

---

# 🔴 Major Problems in Your Code

### ❌ 1. Both classes are identical

<pre class="overflow-visible! px-0!" data-start="263" data-end="370"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">x3</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">200</span><span>,</span><span class="ͼq">2</span><span>) </span><span class="ͼn">+</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>array([</span><span class="ͼn">-</span><span class="ͼq">2</span><span>,</span><span class="ͼq">2</span><span>])</span><br/><span class="ͼt">x4</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randn(</span><span class="ͼq">200</span><span>,</span><span class="ͼq">2</span><span>) </span><span class="ͼn">+</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>array([</span><span class="ͼn">-</span><span class="ͼq">2</span><span>,</span><span class="ͼq">2</span><span>])</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Both are centered at **[-2, 2]**

So your dataset is not separable → Model can’t learn properly.

---

### ❌ 2. Labels are random

<pre class="overflow-visible! px-0!" data-start="502" data-end="550"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>randint(</span><span class="ͼq">0</span><span>,</span><span class="ͼq">2</span><span>,(</span><span class="ͼq">400</span><span>,</span><span class="ͼq">1</span><span>))</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Labels are random and not linked to data.

So model cannot learn any pattern.

---

### ❌ 3. Shuffle problem

You shuffled only `x`, not `y`:

<pre class="overflow-visible! px-0!" data-start="695" data-end="729"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">np</span><span class="ͼn">.</span><span>random</span><span class="ͼn">.</span><span>shuffle(</span><span class="ͼt">x</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Now data and labels don’t match anymore ❌

---

### ❌ 4. Binary Cross Entropy formula is wrong

You used:

<pre class="overflow-visible! px-0!" data-start="838" data-end="867"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">np</span><span class="ͼn">.</span><span>log1p(</span><span class="ͼt">y_hat</span><span>)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Correct formula is:

Loss=−[ylog(y^)+(1−y)log(1−y^)]Loss = -[y log(ŷ) + (1-y) log(1-ŷ)]**L**oss**=**−**[**y**l**o**g**(**y**^)**+**(**1**−**y**)**l**o**g**(**1**−**y**^)]
---------------------------------------------------------------------------------------------------------------

# 🧠 What Is Happening Mathematically?

### 1️⃣ Hypothesis

y^=σ(XW)ŷ = \sigma(XW)**y**^=**σ**(**X**W**)**

### 2️⃣ Loss

J=−[ylog(y^)+(1−y)log(1−y^)]J = -[y log(ŷ) + (1-y) log(1-ŷ)]**J**=**−**[**y**l**o**g**(**y**^)**+**(**1**−**y**)**l**o**g**(**1**−**y**^)]**

### 3️⃣ Gradient

∂J/∂W=XT(y^−y)∂J/∂W = X^T (ŷ - y)**∂**J**/**∂**W**=**X**T**(**y**^****−**y**)**

### 4️⃣ Update Rule

W=W−α×gradientW = W - α × gradient**W**=**W**−**α**×**g**r**a**d**i**e**n**t
--------------------------------------------------------------------------------

# 🎯 Expected Result

Now since classes are separable, you should get:

<pre class="overflow-visible! px-0!" data-start="2643" data-end="2672"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Accuracy ≈ 95% - 100%</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---
