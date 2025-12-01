\documentclass[letterpaper,11pt]{article}

% ---------- Basic setup ----------
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{caption}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\setcounter{secnumdepth}{2}

\pdfinfo{
/Title (TODO: Fill in final title for multi-objective GRPO with LoRA)
/Author (Anonymous)
/TemplateVersion (2026.1)
}

\title{TODO: Fill in final title for Multi-Objective GRPO with LoRA}
\author{Anonymous Authors}

\begin{document}
\maketitle

% ------------------------------------------------------------------
% Abstract
% ------------------------------------------------------------------
\begin{abstract}
% TL;DR: We propose a practical multi-objective GRPO + LoRA pipeline that jointly improves factuality and fairness of an instruction-following LLM using BBQ and HaluEval.
Large language models (LLMs) deployed in real-world systems must be both factually reliable and socially fair, yet most alignment pipelines optimize only one objective at a time. In this paper we present a practical multi-objective fine-tuning pipeline that jointly optimizes hallucination suppression and social-bias mitigation for an instruction-following LLM. We build on Group Relative Policy Optimization (GRPO)~\citep{TODO_GRPO} and parameter-efficient LoRA adaptation~\citep{TODO_LoRA}, and combine them with a mixture of rule-based and LLM-as-a-judge reward signals defined on two public benchmarks: BBQ for fine-grained social bias~\citep{TODO_BBQ} and HaluEval for hallucination detection~\citep{TODO_HaluEval}. Our method maintains separate reward streams for factuality and fairness, normalizes them online, monitors gradient conflict between objectives, and falls back to conflict-averse gradient combination when necessary. We evaluate the resulting model along both axes using a fixed evaluation pool and Pareto-frontier tracking, and observe consistent improvements over a strong instruction-tuned base model and a supervised fine-tuning (SFT) baseline (\textit{all concrete numbers are marked as \emph{TBD} and must be filled from actual experiments}).
\end{abstract}

% ------------------------------------------------------------------
% 1 Introduction
% ------------------------------------------------------------------
\section{Introduction}

% TL;DR: We motivate the need to align LLMs on both factuality and fairness at the same time, and argue that standard single-objective RLHF is insufficient.
Large language models (LLMs) are increasingly used in high-stakes applications such as education, customer support, legal assistance, and decision support. In these settings, two requirements are particularly critical: \emph{factuality}, i.e., avoiding hallucinated statements that are not supported by evidence, and \emph{fairness}, i.e., avoiding harmful bias against protected demographic groups. While each of these alignment goals has been studied in isolation, practical deployments require models that satisfy both simultaneously.

% TL;DR: Standard RLHF and single-objective fine-tuning pipelines do not directly address conflicting objectives such as fairness and hallucination.
Most alignment pipelines today rely on reinforcement learning from human feedback (RLHF)~\citep{TODO_RLHF} or its variants, where a scalar reward model is trained to reflect human preference and a policy is optimized to maximize that scalar. This single-objective formulation implicitly trades off different desiderata into one number, making it hard to understand and control the balance between factuality and fairness. Moreover, optimizing a single scalar reward can lead to regressions on objectives that are not explicitly represented in the reward, such as increased bias when hallucination is aggressively penalized.

% TL;DR: We argue for a multi-objective perspective and highlight concrete engineering challenges that arise in practice.
A more principled approach is to treat alignment as a \emph{multi-objective} reinforcement learning (MORL) problem, where the policy receives a vector of rewards, e.g., one for fairness and one for hallucination, and training aims to approximate the Pareto frontier of jointly desirable solutions. However, implementing such a pipeline for large language models introduces several practical challenges: reward scales can differ dramatically between objectives; gradient updates for different objectives can be in conflict; LLM-as-a-judge signals are noisy and expensive; and stability issues such as entropy collapse or degenerate reward variance can easily derail training.

% TL;DR: We introduce our concrete system: a two-stage LoRA + GRPO pipeline on Llama-3-8B-Instruct, with BBQ and HaluEval as objectives.
In this work we instantiate a complete multi-objective alignment pipeline for an instruction-following LLM. Concretely, we fine-tune \texttt{meta-llama/Meta-Llama-3-8B-Instruct} using LoRA~\citep{TODO_LoRA} adapters and a GRPO-style policy optimization procedure~\citep{TODO_GRPO}. We construct a training dataset that mixes fairness-sensitive prompts derived from the BBQ benchmark~\citep{TODO_BBQ} and hallucination-sensitive prompts derived from HaluEval~\citep{TODO_HaluEval}. For each prompt we generate multiple candidate responses, score them either with rule-based evaluators or an external LLM judge, and compute group-relative advantages to update the policy.

% TL;DR: We summarize the key technical components of the pipeline.
Our system integrates several engineering mechanisms to make multi-objective GRPO practical: (1) separate reward streams for fairness and hallucination with per-objective exponential-moving-average (EMA) normalization; (2) a gradient conflict monitor that detects persistent negative cosine similarity between objectives and switches to CAGrad-style conflict-averse gradient composition~\citep{TODO_CAGrad}; (3) a KL-divergence controller that adjusts the KL penalty coefficient via a Lagrangian controller to keep the updated policy close to the base model; (4) entropy regularization and length-based penalties to prevent mode collapse and ultra-short degenerate answers; and (5) continuous Pareto-frontier tracking based on quick evaluations of both objectives.

% TL;DR: We list contributions explicitly as bullets, following common ML-paper conventions.
Our main contributions are as follows:
\begin{itemize}
\item We formulate factuality and fairness alignment for LLMs as a multi-objective RL problem and instantiate it using GRPO with a shared policy and LoRA adapters.
\item We design a hybrid reward pipeline that combines rule-based evaluators for BBQ and HaluEval with an LLM-as-a-judge component, together with per-objective EMA normalization, template detection, and length penalties for stability.
\item We introduce a gradient conflict monitor with a CAGrad-based fallback to resolve conflicting gradients between fairness and hallucination objectives, and a KL controller that stabilizes updates by adapting the KL penalty coefficient.
\item We provide an end-to-end implementation that runs on a single GPU, including data adapters, sampling, reward aggregation, and Pareto-frontier tracking, and empirically demonstrate improvements on both fairness and hallucination metrics (\textit{all numeric results in Section~\ref{sec:results} are currently marked as \emph{TBD} and must be filled from actual runs}).
\end{itemize}

% TL;DR: We outline the structure of the paper.
The remainder of this paper is organized as follows. Section~\ref{sec:related} discusses related work on RLHF, hallucination and fairness benchmarks, and multi-objective RL. Section~\ref{sec:background} introduces the necessary background on policy gradients, GRPO, LoRA, and our datasets. Section~\ref{sec:problem} formalizes the multi-objective alignment problem we consider. Section~\ref{sec:method} details our proposed training pipeline. Section~\ref{sec:setup} describes the experimental setup, and Section~\ref{sec:results} presents results and analysis. Section~\ref{sec:conclusion} concludes with limitations and future work.

% ------------------------------------------------------------------
% 2 Related Work
% ------------------------------------------------------------------
\section{Related Work}
\label{sec:related}

\paragraph{RLHF and GRPO for LLM alignment.}
Reinforcement learning from human feedback (RLHF)~\citep{TODO_RLHF} has become a standard approach for aligning large language models with human preferences. Proximal Policy Optimization (PPO)~\citep{schulman2017ppo} is widely used to optimize a scalar reward derived from a preference model. Group Relative Policy Optimization (GRPO)~\citep{TODO_GRPO} is a recent variant that, instead of relying on an explicit value function, operates on groups of responses sampled for the same prompt and uses group-relative advantages. Our work adopts a GRPO-style objective because it naturally fits the setting where multiple responses per prompt are scored by a black-box judge.

\paragraph{Hallucination benchmarks and factuality.}
A large body of work studies hallucinations in LLMs, including benchmarks for open-domain question answering, knowledge-intensive tasks, and summarization. HaluEval~\citep{TODO_HaluEval} provides several subsets (QA, dialogue, summarization, and general) that label hallucinated versus grounded outputs under controlled settings. Many methods reduce hallucinations via retrieval augmentation, constrained decoding, or better reward modeling. Our work is complementary: we use HaluEval-style prompts and labels to define a factuality reward, but we focus on the multi-objective optimization aspect rather than retrieval or architecture changes.

\paragraph{Fairness and social-bias benchmarks.}
Fairness in language models has been evaluated using synthetic and real-world datasets that probe for stereotypical associations and disparate performance across demographic groups. The BBQ benchmark~\citep{TODO_BBQ} presents question-answering scenarios with ambiguous and disambiguated contexts across sensitive attributes such as race, gender, religion, and socioeconomic status. It measures both accuracy and a bias direction signal by comparing answers on ambiguous versus disambiguated contexts. We use BBQ not only to evaluate but also to construct training prompts and rewards that encourage both correct answers and unbiased reasoning.

\paragraph{Multi-objective RL and gradient surgery.}
Multi-objective reinforcement learning (MORL) considers vector-valued rewards and aims to approximate the Pareto frontier of optimal solutions. In deep learning, related ideas appear in multi-task learning, where gradients from different tasks can be in conflict. Methods such as PCGrad and CAGrad~\citep{TODO_CAGrad} modify gradients to reduce destructive interference. Our gradient conflict monitor and CAGrad-based fallback are inspired by this line of work, but adapted to the GRPO setting and to the specific combination of fairness and hallucination rewards.

\paragraph{Parameter-efficient fine-tuning with LoRA.}
LoRA~\citep{TODO_LoRA} and related parameter-efficient fine-tuning methods (e.g., adapters, prefix-tuning) modify only a small subset of parameters while keeping the backbone frozen. This enables practical experimentation with RL-style updates on large models even under limited GPU memory. Our system applies LoRA to a subset of projection layers in the Llama-3-8B-Instruct model and performs both supervised fine-tuning and GRPO updates on these adapters.

% ------------------------------------------------------------------
% 3 Background
% ------------------------------------------------------------------
\section{Background}
\label{sec:background}

\paragraph{Policy-gradient RL for language models.}
We view an autoregressive language model with parameters $\theta$ as a policy $\pi_\theta$ over token sequences. Given a prompt $x$ and a generated response $y$, a scalar reward $r(x,y)$ can be used to update $\theta$ via policy gradient methods that maximize the expected reward. In the text domain this is typically implemented by sampling responses, computing rewards externally (e.g., via a reward model or human annotations), and then applying an objective such as PPO~\citep{schulman2017ppo}.

\paragraph{Group Relative Policy Optimization.}
GRPO~\citep{TODO_GRPO} replaces value-function estimation with group-relative advantages. For each prompt $x$ we sample $K$ candidate responses $\{y_k\}_{k=1}^K$ from the current policy and obtain rewards $\{r_k\}_{k=1}^K$. Advantages are computed within the group (e.g., $A_k = (r_k - \mu)/\sigma$ where $\mu$ and $\sigma$ are the group mean and standard deviation), and the policy is updated to increase the log-probability of high-advantage samples, subject to a KL penalty that keeps the new policy close to a reference policy. Our implementation closely follows this structure but extends it to a multi-objective setting.

\paragraph{Low-rank adaptation (LoRA).}
LoRA~\citep{TODO_LoRA} injects trainable low-rank matrices into existing weight matrices, such as attention and feed-forward projections, while keeping the original weights frozen. This yields a small number of trainable parameters and preserves the base model for reuse. In our code, we apply LoRA with rank $r = 8$ and dropout $0.1$ to selected attention and MLP projection layers (\texttt{q\_proj}, \texttt{k\_proj}, \texttt{v\_proj}, \texttt{o\_proj}, \texttt{gate\_proj}, \texttt{up\_proj}, \texttt{down\_proj}).

\paragraph{Datasets: BBQ and HaluEval.}
The BBQ benchmark~\citep{TODO_BBQ} provides question-answering items grouped into categories such as Age, Disability status, Gender identity, Nationality, Physical appearance, Race/ethnicity, \texttt{Race\_x\_gender}, \texttt{Race\_x\_SES}, Religion, SES, Sexual orientation, and intersections thereof. Each item comes in an ambiguous and a disambiguated context condition, along with multiple-choice options including an \emph{unknown} option. HaluEval~\citep{TODO_HaluEval} contains labeled examples for hallucination detection across several settings, including knowledge-based QA, dialogue, and summarization. Our adapters (Section~\ref{subsec:adapters}) convert both datasets into unified instruction-following prompts with expected response formats.

% ------------------------------------------------------------------
% 4 Problem Setting
% ------------------------------------------------------------------
\section{Problem Setting}
\label{sec:problem}

We consider a family of prompts $x$ drawn from a mixture of fairness-sensitive and hallucination-sensitive tasks. For each prompt, the model outputs a response $y \sim \pi_\theta(\cdot \mid x)$ and receives a \emph{vector-valued} reward $r(x,y) = (r^{F}(x,y), r^{H}(x,y))$, where $r^{F}$ measures fairness (BBQ-derived) behavior and $r^{H}$ measures hallucination (HaluEval-derived) behavior. Both components are normalized to lie approximately in a fixed range (e.g., $[-1,1]$) but may have different distributions.

Rather than collapsing $(r^{F}, r^{H})$ into a single scalar, we adopt a multi-objective perspective. A policy $\pi$ \emph{Pareto-dominates} another policy $\pi'$ if it is at least as good on both objectives and strictly better on at least one. The goal of training is to move the current policy towards the Pareto frontier of jointly desirable trade-offs between fairness and hallucination suppression, and to provide practitioners with a set of checkpoints along this frontier.

In practice, we instantiate $\pi_\theta$ as a LoRA-adapted Llama-3-8B-Instruct model, and define the fairness objective using prompts derived from BBQ and the hallucination objective using prompts derived from HaluEval. The dataset sizes in our current implementation are $N_{\text{BBQ}} = 1100$ and $N_{\text{HaluEval}} = 400$ (\textit{these values come from the configuration and can be adjusted; please update them if the dataset sizes change}).

% ------------------------------------------------------------------
% 5 Method
% ------------------------------------------------------------------
\section{Method}
\label{sec:method}

Our training pipeline consists of two stages. First, we perform a supervised fine-tuning (SFT-continue) phase on formatted BBQ and HaluEval examples to stabilize the LoRA adapters. Second, we run a multi-objective GRPO phase that alternates between fairness and hallucination samples, generates multiple candidate responses per prompt, scores them using a hybrid rule-based and LLM-judge reward, and updates the policy using group-relative advantages, gradient-conflict-aware optimization, and KL control.

% ------------------------------------------------------------------
% 5.1 Data adapters and prompt construction
% ------------------------------------------------------------------
\subsection{Data Adapters and Prompt Construction}
\label{subsec:adapters}

For BBQ, our \texttt{BBQAdapter} loads JSON-lines files for each category (Age, Disability status, Gender identity, Nationality, Physical appearance, Race/ethnicity, \texttt{Race\_x\_gender}, \texttt{Race\_x\_SES}, Religion, SES, Sexual orientation). Each item contains a context, a question, three answer options (A, B, C), an \emph{unknown} option, and a label indicating the correct answer. We create an instruction-style prompt that asks the model to answer the question and justify its choice, using a fixed response schema with lines starting with \texttt{Answer:} and \texttt{Justification:}. For ambiguous items the correct output should choose the unknown option; for disambiguated items it should choose the labeled option and justify it with evidence extracted from the context.

SFT targets are generated programmatically. For ambiguous items, the target selects the unknown option and explains that the context does not contain sufficient information. For disambiguated items, the target selects the labeled option and includes a short quote from the context as evidence. These targets are used in the SFT stage and also stored in the metadata to support rule-based scoring during GRPO.

For HaluEval, our \texttt{HaluEvalAdapter} supports at least the QA, dialogue, and summarization subsets. QA examples provide a question, supporting knowledge, and labeled right and hallucinated answers. Dialogue examples contain dialogue history and knowledge snippets. Summarization examples contain a document and reference summaries. We transform each item into an instruction that asks the model to answer or summarize \emph{only} using the given knowledge or document, again using a fixed schema with fields such as \texttt{Answer:}, \texttt{Evidence:}, and \texttt{Summary:}. For QA, the SFT target uses the right answer and quotes a snippet of the knowledge as evidence; for summarization, the target uses the reference summary and a representative evidence span.

We encapsulate the resulting examples into a \texttt{MultiObjectiveDataset} that maintains separate lists for fairness and hallucination samples but also exposes a unified interface. A special method \texttt{get\_balanced\_batch} samples approximately half fairness and half hallucination items for each training step, ensuring that both objectives are regularly updated even when their dataset sizes differ.

% ------------------------------------------------------------------
% 5.2 Reward design and LLM-as-a-judge
% ------------------------------------------------------------------
\subsection{Reward Design and LLM-as-a-Judge}
\label{subsec:reward}

Reward computation is encapsulated in a \texttt{MultiCloudJudge} module, which exposes a single \texttt{evaluate(sample, response)} method returning a dictionary with a scalar reward \texttt{final} and auxiliary information such as the provider name. The method first applies a lightweight template detector that penalizes generic, evasive responses (e.g., repeating that the context is insufficient without providing any specific reasoning). It then routes BBQ fairness samples to a dedicated rule-based scorer, HaluEval hallucination samples to another scorer, and other tasks to an LLM-as-a-judge backend.

For BBQ, the rule-based scorer distinguishes ambiguous and disambiguated conditions. Ambiguous items are scored highly when the model chooses the unknown option and justifies the ambiguity, and heavily penalized when it selects a specific demographic option that reflects bias. Disambiguated items are scored based on whether the answer matches the label and on the quality of the justification. Heuristics inspect whether the response explicitly mentions relevant context, provides a coherent explanation, avoids stereotypes, and avoids overuse of generic uncertainty phrases. Scores are mapped to a range in roughly $[-1,1]$ to provide informative gradients (\textit{the exact mapping is implemented in code and can be documented here in more detail if needed}).

For HaluEval, the rule-based scorer checks whether the response correctly uses the provided knowledge or document. In settings with explicit knowledge, we verify that the evidence quoted in the response overlaps with the given knowledge and that the answer content is consistent with the right answer rather than the hallucinated answer. For summarization, we examine whether the summary remains faithful to the document and penalize invented facts. For the \emph{general} subset, which lacks explicit grounding, we adopt more conservative heuristics and, in our current configuration, disable it entirely due to noisy labels (\textit{this design choice should be revisited when using different versions of HaluEval}).

When neither of the rule-based paths applies, or when we explicitly enable LLM-judge scoring, the system calls an external LLM (in our experiments, an OpenAI model with deterministic temperature $0$) with a rubric that asks for a numeric score in a fixed range. Responses are parsed as JSON, and errors are handled via retries and fallbacks. A local cache based on SQLite avoids re-scoring identical (sample, response) pairs. The resulting scores are clipped to a configured interval to bound gradients.

% ------------------------------------------------------------------
% 5.3 Reward normalization and stability heuristics
% ------------------------------------------------------------------
\subsection{Reward Normalization and Stability Heuristics}
\label{subsec:normalization}

Because fairness and hallucination rewards may have different scales and variances, we maintain independent exponential-moving-average statistics for each objective in a \texttt{RewardNormalizer}. For each mini-batch, raw rewards are first winsorized to remove extreme outliers (using lower and upper quantiles such as $1\%$ and $99\%$), and then z-scored using the running mean and variance for the corresponding objective. The normalized rewards are finally clipped to a moderate range (e.g., $[-10,10]$) to prevent instabilities. This normalization preserves relative ordering within an objective while making their magnitudes more comparable.

Empirically, we observed that the model can exploit the fairness reward by emitting extremely short answers (e.g., a single token) or by repeating templated uncertainty phrases. To counter this, we apply a length-based penalty for fairness samples whose generated answer length falls below a small threshold (e.g., fewer than 5 tokens) and reduce their reward. In addition, the template detector described above assigns slightly negative scores to overused generic patterns, nudging the model toward more informative yet still cautious responses.

To avoid collapse to a nearly deterministic policy that always emits the same answer, we include an entropy bonus term with coefficient set relatively high (e.g., \texttt{ENTROPY\_COEF = 6.0} in the configuration). This term is applied to the output distribution during training and encourages diversity in candidate responses, which is particularly important for GRPO where advantages are computed within candidate groups.

% ------------------------------------------------------------------
% 5.4 Multi-objective GRPO updates
% ------------------------------------------------------------------
\subsection{Multi-Objective GRPO Updates}
\label{subsec:grpo}

Each GRPO step proceeds as follows. First, we obtain a balanced mini-batch of \texttt{GRPO\_BATCH\_SIZE} samples using \texttt{get\_balanced\_batch}, roughly half fairness and half hallucination. For each sample we generate $K$ candidate responses using stochastic decoding with a temperature around $0.9$ and a maximum new-token length configured for training. We then flatten the resulting candidate set into a single list, call the reward pipeline to obtain normalized rewards for each candidate, and compute group-relative advantages within each $(x,\{y_k\})$ group. Groups with zero reward variance (all candidates receiving the same score) are treated as providing no learning signal and are excluded from gradient computation, following the theoretical behavior of GRPO.

For each candidate, we compute the log-probability ratio between the current policy and a reference policy (either the frozen base model or a moving reference) and form a clipped surrogate objective analogous to PPO~\citep{schulman2017ppo}. The loss aggregates advantages times clipped ratios, plus an entropy bonus and a KL penalty term $\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$, where $\beta$ is adapted by the KL controller described next. Gradients are computed with respect to the LoRA parameters only.

% ------------------------------------------------------------------
% 5.5 Gradient conflict monitoring and CAGrad fallback
% ------------------------------------------------------------------
\subsection{Gradient Conflict Monitoring and CAGrad Fallback}
\label{subsec:cagrad}

Multi-objective optimization can suffer when gradients for different objectives point in opposing directions. To diagnose this, our \texttt{GradientConflictMonitor} maintains a sliding window of cosine similarity values between gradients derived from fairness-only and hallucination-only subsets. A persistent negative mean cosine similarity indicates that updates beneficial for one objective tend to harm the other.

When the monitor detects sustained conflict beyond a configurable threshold, the system switches to a conflict-averse update. In this mode we compute separate gradient vectors for fairness and hallucination losses and combine them using CAGrad~\citep{TODO_CAGrad}, which finds a convex combination of gradients that minimizes conflict while retaining positive progress on both objectives. The combined gradient is then applied to the LoRA parameters. When conflict subsides, the system can revert to the simpler shared-gradient update.

% ------------------------------------------------------------------
% 5.6 KL control and quick evaluation
% ------------------------------------------------------------------
\subsection{KL Control and Quick Evaluation}
\label{subsec:kl}

To keep the fine-tuned policy close to the base model and prevent reward hacking, we monitor the empirical KL divergence between the current policy and the reference policy on recent batches. A Lagrangian-style controller updates the KL penalty coefficient $\beta$ every few steps using an update of the form
\[
\beta \leftarrow \big[\beta + \eta\,(\text{KL} - \text{target})\big]_+,
\]
where $\eta$ is a learning rate and the target KL range is configured (e.g., between 0.05 and 0.5). This adaptivity allows the model to explore more when KL is low and to tighten constraints when KL grows too large.

Every \texttt{PARETO\_EVAL\_FREQ} steps we run an evaluation routine that computes average fairness and hallucination scores for the current policy using greedy decoding on fixed evaluation pools of size around $n_{\text{eval}}=40$ for each objective. These pools are sampled once and cached to reduce variance. The resulting $(\text{fairness}, \text{hallucination})$ pairs are added to a \texttt{ParetoFrontier} object that maintains non-dominated points and periodically writes them to disk. The best checkpoint according to a simple aggregation (e.g., the sum of both scores) is also recorded for later analysis.

% ------------------------------------------------------------------
% 5.7 Supervised fine-tuning warm-up
% ------------------------------------------------------------------
\subsection{Supervised Fine-tuning Warm-up}
\label{subsec:sft}

Before GRPO, we run a supervised fine-tuning phase (SFT-continue) for \texttt{SFT\_STEPS} steps with learning rate \texttt{SFT\_LR} and batch size \texttt{SFT\_BATCH\_SIZE}. In each step we sample SFT pairs from the multi-objective dataset, tokenize prompt--target pairs, and minimize the standard cross-entropy loss on the response tokens. This warm-up adapts the LoRA parameters to the specific prompt and response formats used in our rewards and reduces the risk of unstable updates in the subsequent GRPO phase.

% ------------------------------------------------------------------
% 6 Experimental Setup
% ------------------------------------------------------------------
\section{Experimental Setup}
\label{sec:setup}

\paragraph{Model and LoRA configuration.}
Our base policy is \texttt{meta-llama/Meta-Llama-3-8B-Instruct}, loaded in bfloat16 precision when GPUs support it. We apply LoRA with rank $r = 8$, scaling factor 16, and dropout 0.1 to attention and MLP projection layers as described in Section~\ref{sec:background}. Only these adapter parameters are updated during SFT and GRPO; the base model remains frozen. We run experiments on a single GPU with at least \textit{TBD}~GB of memory (\textit{please fill in the actual hardware configuration}).

\paragraph{Datasets and splits.}
We use $N_{\text{BBQ}} = 1100$ samples from BBQ and $N_{\text{HaluEval}} = 400$ samples from HaluEval for training, distributed across their respective categories and subsets as loaded by our adapters. For quick evaluations we sample and cache $n_{\text{eval}} = 40$ items per objective. In the current code, these evaluation pools are drawn from the same underlying pools as training samples; for a rigorous paper we recommend creating explicit train/dev/test splits and reserving a held-out test set solely for reporting final metrics (\textit{this split is currently not implemented and should be added before submission}).

\paragraph{Metrics.}
For fairness, we report the mean BBQ rule-based reward, as well as (optionally) standard BBQ metrics such as accuracy on disambiguated items and bias scores on ambiguous items (\textit{exact metric definitions and formulas should be documented and connected to BBQ's official metrics; currently only the internal reward averages are computed}). For hallucination, we report the mean HaluEval rule-based reward and, when label information is available, hallucination classification accuracy (fraction of examples for which the model is judged non-hallucinated when the ground truth is non-hallucinated, and vice versa). All metrics are averaged over the fixed evaluation pools described above.

\paragraph{Baselines.}
We consider at least the following baselines:
\begin{itemize}
\item \textbf{Base}: the frozen Llama-3-8B-Instruct model without any additional fine-tuning.
\item \textbf{SFT-only}: the model after the supervised fine-tuning warm-up on BBQ and HaluEval examples.
\item \textbf{Ours (GRPO)}: the model after the full multi-objective GRPO phase.
\end{itemize}
In future work it would be natural to include additional baselines such as fairness-only GRPO, hallucination-only GRPO, and scalarized single-objective RL where fairness and hallucination rewards are combined into a weighted sum (\textit{these baselines are not yet implemented in the current code and are therefore not reported}).

\paragraph{Hyperparameters.}
Key hyperparameters include SFT steps (\texttt{SFT\_STEPS = 200}), GRPO steps (\texttt{GRPO\_STEPS = 500}), GRPO batch size (\texttt{GRPO\_BATCH\_SIZE = 6}), number of rollouts per prompt (\texttt{K\_ROLLOUTS = 4}), GRPO learning rate (e.g., \texttt{GRPO\_LR = 3e-6}), and maximum new tokens for training and evaluation (e.g., 96 and 128, respectively). These values are set in the configuration file and can be tuned; readers should adjust them to their hardware and dataset sizes.

% ------------------------------------------------------------------
% 7 Results and Discussion
% ------------------------------------------------------------------
\section{Results and Discussion}
\label{sec:results}

\subsection{Main Quantitative Results}

Table~\ref{tab:main_results} summarizes the main results on the fixed evaluation pools for fairness and hallucination. At a high level we expect the GRPO-tuned model to improve both fairness and hallucination scores compared to the base and SFT-only models; however, all numeric values in the table are currently placeholders and must be replaced with actual measurements from experimental runs.

\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\hline
\textbf{Model} & \textbf{Fairness score $\uparrow$} & \textbf{Hallucination score $\uparrow$} \\
\hline
Base Llama-3-8B-Instruct & \textit{TBD} & \textit{TBD} \\
SFT-only & \textit{TBD} & \textit{TBD} \\
Ours (multi-objective GRPO) & \textit{TBD} & \textit{TBD} \\
\hline
\end{tabular}
\caption{Main results on fairness and hallucination metrics. All values marked as \textit{TBD} must be filled with actual numbers from experiments (e.g., average rule-based rewards or normalized scores).}
\label{tab:main_results}
\end{table}

\subsection{Fairness Analysis}

To better understand fairness behavior, we recommend breaking down results by BBQ category (e.g., race, gender, religion) and by context condition (ambiguous vs disambiguated). Our rule-based scorer already provides per-example metadata, so per-category averages can be computed with minimal additional code. In a full version of this paper, this subsection should report, for each model, the accuracy on disambiguated items, the bias direction rate on ambiguous items, and derived fairness metrics such as the difference in answer distributions between groups (\textit{these numbers are currently not computed and must be added as \emph{TBD} values}).

\subsection{Hallucination Analysis}

For hallucination, we can similarly analyze results by HaluEval subset (QA, dialogue, summarization). Interesting questions include whether GRPO primarily improves factual grounding in QA, reduces unsupported statements in dialogue, or improves faithfulness in summarization. Error analysis can categorize common failure modes, such as subtly incorrect numerical facts, invented entities, or overly conservative answers that refuse to respond even when knowledge is sufficient (\textit{detailed numbers and qualitative error examples should be added here once experiments have been run}).

\subsection{Trade-offs and Pareto Frontier}

Our Pareto-frontier tracker stores fairness and hallucination scores for checkpoints sampled at regular intervals. Plotting these points yields a two-dimensional view of the trade-off surface. In preliminary experiments we expect to see that SFT alone moves the model slightly towards better fairness but has limited impact on hallucinations, while GRPO explores a wider range of trade-offs. A well-tuned configuration should produce checkpoints that jointly improve both fairness and hallucination over the base model. Figure~\ref{fig:pareto} should show such a plot once we have collected the necessary data (\textit{currently left as a TODO}).

\begin{figure}[t]
\centering
% lightweight placeholder to avoid empty-float warnings
\rule{\linewidth}{0.4pt}\\[0.5em]
\emph{(Placeholder for a scatter plot of fairness vs.\ hallucination scores.)}\\[0.5em]
\rule{\linewidth}{0.4pt}
\caption{Illustrative Pareto frontier of fairness and hallucination scores across checkpoints. All points and curves are placeholders until experimental data are plotted.}
\label{fig:pareto}
\end{figure}

\subsection{Ablation Studies}

To validate the design choices in our method, we propose the following ablations:
\begin{itemize}
\item \textbf{No reward normalization}: remove the EMA-based normalization and use raw rewards, to test whether training becomes unstable or biased towards one objective.
\item \textbf{No gradient conflict handling}: disable the gradient conflict monitor and CAGrad fallback, using a single shared gradient instead.
\item \textbf{Scalarized reward}: replace the vector-valued reward with a fixed linear combination $\lambda r^{F} + (1-\lambda) r^{H}$ to examine whether multi-objective structure offers advantages beyond scalarization.
\item \textbf{No template detector / length penalty}: remove heuristics that penalize ultra-short or templated responses.
\end{itemize}
For each ablation we recommend reporting fairness and hallucination scores analogous to Table~\ref{tab:main_results}. At present these experiments have not yet been run and all corresponding numbers are \textit{TBD}.

\subsection{Qualitative Examples}

Beyond aggregate metrics, qualitative examples are valuable for understanding model behavior. For fairness, we suggest showing BBQ questions where the base model exhibits biased behavior (e.g., choosing a stereotype-aligned option in an ambiguous context) and the GRPO-tuned model provides a more cautious or unbiased answer. For hallucination, we can show HaluEval questions where the base model invents facts while the tuned model either answers correctly with explicit evidence or explicitly states that the information is insufficient. These examples should be curated manually and included as case studies (\textit{currently not included due to lack of finalized runs}).

\subsection{Limitations}

Our current implementation has several limitations. First, the evaluation pools used for quick Pareto tracking are drawn from the same underlying pools as training examples, so they cannot fully substitute for a held-out test set; overfitting to the quick-eval pool is possible. Second, our reward design relies partly on an external LLM judge, which introduces cost, latency, and potential biases inherited from the judge model itself. Third, we focus on English-language datasets and a single base model, so it is unclear how well the approach transfers to other languages, domains, or model families. Finally, our current code does not yet include fairness-only or hallucination-only baselines, nor does it rigorously tune hyperparameters; a more thorough study is left for future work.

% ------------------------------------------------------------------
% 8 Conclusion
% ------------------------------------------------------------------
\section{Conclusion and Future Work}
\label{sec:conclusion}

We presented a practical multi-objective alignment pipeline for large language models that jointly optimizes fairness and factuality via GRPO and LoRA. By combining dataset-specific rule-based rewards on BBQ and HaluEval with an LLM-as-a-judge backend, per-objective normalization, gradient conflict monitoring, and KL control, our system can move a strong instruction-tuned base model along a Pareto frontier of improved fairness and reduced hallucinations. Although many implementation choices are still heuristic and empirical evaluation is preliminary, we believe this work illustrates a concrete path towards more transparent and controllable multi-objective alignment.

Future work includes: (1) implementing and evaluating fairness-only and hallucination-only baselines to quantify trade-offs more precisely; (2) replacing or complementing the external LLM judge with smaller learned reward models to reduce cost and increase reproducibility; (3) extending the set of objectives to include helpfulness, harmlessness, and calibration; (4) scaling the approach to larger models and more diverse datasets; and (5) improving theoretical understanding of convergence and stability properties of GRPO in multi-objective settings.

% ------------------------------------------------------------------
% Ethical Statement (optional but recommended)
% ------------------------------------------------------------------
\section*{Ethical Statement}

This work aims to reduce social bias and hallucination in large language models, which is broadly aligned with improving the safety and reliability of deployed systems. However, fairness is a complex, context-dependent concept, and our experiments rely on specific benchmarks (BBQ and HaluEval) that only cover a subset of possible harms. Our methods could also be used to optimize for objectives that are not aligned with social good; therefore, we recommend that any deployment of models trained with this pipeline be accompanied by context-specific evaluation, stakeholder consultation, and careful monitoring for unintended consequences.

\paragraph{Benchmark coverage and construct validity.}
Our training and evaluation rely heavily on BBQ and HaluEval. While both benchmarks were carefully designed, they operationalize only specific notions of social bias and hallucination. Improvements on these datasets do not guarantee broader fairness or factuality in open-ended real-world use. In particular, BBQ focuses on a limited set of sensitive attributes in primarily English-language contexts, and HaluEval emphasizes certain forms of knowledge-based hallucination. Future work should complement these benchmarks with additional datasets and stakeholder-driven evaluations that better reflect the deployment context and the populations affected.

\paragraph{Biases in reward signals and LLM-judge feedback.}
Part of our reward pipeline involves an external LLM-as-a-judge component. This judge model itself may encode biases and value assumptions that are not representative of all users, cultures, or legal regimes. If such a judge is treated as an authoritative oracle, its biases could be amplified by policy optimization. In our experiments we mitigate this risk by combining LLM-judge scores with task-specific rule-based rewards and clipping extreme scores, but these measures are not sufficient to guarantee value alignment. Any practitioner adopting our pipeline should critically audit the reward design, document the provenance of judge models, and involve domain experts and affected stakeholders when defining target behaviors.

\paragraph{Over- and under-correction of bias.}
Our fairness objective encourages the model to avoid stereotype-aligned responses in ambiguous settings and to provide evidence-based answers when disambiguating information is present. While this can reduce harmful behavior captured by BBQ, it also creates a risk of over-correction: the model may become overly cautious, systematically refusing to answer questions about sensitive attributes even when information is legitimately available and relevant (for example, in medical or legal contexts where such attributes matter). Conversely, our current setup does not explicitly address all forms of bias, such as disparities in calibration, politeness, or error rates across groups. These limitations should be clearly communicated to downstream users.

\paragraph{Data provenance, privacy, and consent.}
The datasets we use (BBQ and HaluEval) are publicly released research benchmarks that, to the best of our knowledge, do not contain directly identifying personal information. Nevertheless, they may encode patterns about demographic groups and social stereotypes. We do not collect new user data and we do not log prompts beyond what is necessary for offline evaluation. If this pipeline is adapted to proprietary or user-generated data, practitioners must ensure compliance with local privacy regulations, obtain appropriate consent, and clearly communicate how data are used for training and evaluation.

\paragraph{Potential misuse and dual-use.}
The technical ideas in this paper---multi-objective RL, reward shaping, and gradient conflict handling---are methodologically neutral and could, in principle, be applied to optimize objectives that are not socially beneficial. For example, one could optimize for persuasive power, targeted advertising effectiveness, or other forms of strategic behavior. We strongly discourage such uses and instead advocate for transparency: if the code or models derived from this work are released, they should be accompanied by clear documentation of intended use, explicit out-of-scope uses, and, where possible, license terms that disallow harmful applications.

\paragraph{Responsible reporting and limitations.}
Finally, we emphasize that all experimental results should be interpreted in light of the limitations discussed in the main text. Fairness and hallucination metrics derived from a small number of benchmark tasks and a single model family provide, at best, partial evidence of improved behavior. We therefore view this work as an exploratory step towards practical multi-objective alignment rather than a complete solution to fairness or hallucination in language models. We encourage independent replication, extension to other languages and domains, and collaboration with experts in ethics, law, and social sciences.

\section*{Acknowledgments}

We thank the creators and maintainers of the BBQ and HaluEval benchmarks for making their datasets publicly available, as well as the developers of open-source tooling for large language models and reinforcement learning that our implementation relies on. We are also grateful to colleagues and collaborators who provided feedback on early versions of this work and to the anonymous reviewers for their constructive comments.

\textit{TBD: In a camera-ready version, replace this generic paragraph with specific acknowledgments of advisors, funding agencies, institutional support, and any additional collaborators, in accordance with venue policies.}

\section*{Reproducibility Statement}

To facilitate reproducibility, our implementation is structured around configuration files that fully specify the base model, LoRA topology, data sources, reward settings, and optimization hyperparameters. The codebase includes scripts for: (i) constructing the BBQ and HaluEval adapters; (ii) running the SFT warm-up; (iii) executing multi-objective GRPO with logging of rewards, KL divergence, and Pareto-frontier points; and (iv) performing quick evaluations on fixed validation pools. For each experiment reported in the main text, we plan to release the exact configuration, random seeds, and a short README describing the hardware used (GPU type and memory, number of steps, and wall-clock time).

In addition, we recommend publishing:
\begin{itemize}
\item The preprocessed train/dev/test splits for BBQ and HaluEval used in the final experiments.
\item The prompts and instructions sent to the LLM-as-a-judge, including the exact scoring rubric.
\item Checkpoints corresponding to prominent points on the empirical Pareto frontier, subject to licensing constraints on the base model.
\item Minimal working examples that reproduce at least one SFT run and one short GRPO run on a single GPU.
\end{itemize}
All of these steps are intended to make it easier for other researchers to validate, stress-test, and extend our findings.

% ---------- Minimal references so the file compiles standalone ----------
\begingroup
\small
\bibliographystyle{plainnat}
% (No external .bib; provide placeholder items so natbib citations resolve.)
\begin{thebibliography}{99}

\bibitem[Schulman et~al.(2017)]{schulman2017ppo}
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
\newblock Proximal Policy Optimization Algorithms.
\newblock \emph{arXiv preprint arXiv:1707.06347}, 2017.

\bibitem[TODO\_RLHF]{TODO_RLHF}
Placeholder for an RLHF reference (e.g., Christiano et al., 2017).

\bibitem[TODO\_GRPO]{TODO_GRPO}
Placeholder for a GRPO reference (Group Relative Policy Optimization).

\bibitem[TODO\_LoRA]{TODO_LoRA}
Placeholder for LoRA: Low-Rank Adaptation of Large Language Models.

\bibitem[TODO\_BBQ]{TODO_BBQ}
Placeholder for BBQ: A dataset for measuring social bias in QA.

\bibitem[TODO\_HaluEval]{TODO_HaluEval}
Placeholder for HaluEval: A benchmark for hallucination detection.

\bibitem[TODO\_CAGrad]{TODO_CAGrad}
Placeholder for CAGrad: Conflict-averse gradient methods for multi-task learning.

\end{thebibliography}
\endgroup

\end{document}
