---
title: "Evaluating the Impact of Retrieval Strategies on the Accuracy of Retrieval-Augmented Generation Systems"
subtitle: "CS391: Case Studies in Machine Learning, The University of Texas at Austin"
date: "November 25, 2026"
fontsize: 11pt
geometry: margin=1in
header-includes:
  - \usepackage{float}
  - \usepackage{booktabs}
  - \hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}
---

# Abstract

Retrieval-augmented generation can supply external evidence to a language model, but better retrieval need not produce better answers. This study compares BM25 keyword retrieval, pretrained semantic retrieval, and a supervised logistic regression relevance reranker while holding the generation model, questions, corpus, and scoring procedures fixed. A reproducible Natural Questions prototype contains 500 training, 50 validation, and 150 test questions, with a shared corpus of 696 deduplicated source passages. The reranker combines BM25 score, semantic similarity, question-passage word overlap, and passage length. Ten conditions comprise an LLM-only baseline and each retrieval method at K=3, 5, and 10, yielding 1,500 saved test responses from GPT-4.1-mini. Every RAG condition exceeded the baseline's {{llm_only_0_mean_em}}% Exact Match and {{llm_only_0_mean_f1}} token F1. Reranker K=3 achieved the highest observed Exact Match, {{reranker_3_mean_em}}%, and F1, {{reranker_3_mean_f1}}. Its paired F1 improvement over semantic K=3 was {{pair2_delta_f1}}, with a pointwise 95% bootstrap interval of [{{pair2_f1_ci_low}}, {{pair2_f1_ci_high}}]. Corresponding differences at K=5 and K=10 had intervals spanning zero. Larger K consistently increased input tokens but produced mixed accuracy changes. Review of 24 deliberately selected questions identified answer-format penalties, ambiguous references, and source-present abstention. These findings support evaluating retrieval, generation, and lexical scoring separately. The selected answer-bearing corpus, limited test sample, and incomplete relevance annotations constrain generalization.

# 1. Introduction

Language models are attractive question-answering interfaces because they can generate short responses without requiring users to formulate database queries. Their parameters can encode substantial relational knowledge, although access varies across facts and relations [@lama]. For an application that needs traceable evidence, a fluent answer alone is insufficient: users also need to know which documents were selected and whether those documents support the requested claim. Retrieval-augmented generation introduces an external evidence source rather than relying exclusively on parametric memory [@rag].

This architecture creates a practical design question. Should a system select passages through lexical matching, embedding similarity, or supervised reranking? BM25 rewards overlapping terms and remains an important retrieval baseline [@bm25]. Dense encoders can match differently worded questions and passages [@dpr]. A reranker can combine signals from both, but its additional computation and supervision are useful only if improved ordering translates into better downstream responses. These distinctions matter even when the language model itself remains unchanged.

The project therefore studies retrieval as an experimentally controlled component. It uses the same passage corpus, test questions, generation snapshot, instructions, and answer metrics across conditions. A small logistic model is trained to combine four observable question-passage features, allowing a supervised machine-learning contribution without training the language model or a neural cross-encoder. This setting makes the comparison feasible for a course project while retaining separate training, validation, and final evaluation stages.

A modest fixed corpus also makes passage-level disagreements inspectable: the analysis can trace a changed response to saved source IDs without reconstructing an evolving external search index.

Four questions organize the investigation. RQ1 asks how RAG compares with an LLM-only baseline in answer accuracy. RQ2 asks how BM25, semantic retrieval, and the trained reranker compare in source recovery and downstream quality. RQ3 asks how K=3, 5, and 10 affect Exact Match, token F1, token consumption, and recorded latency. RQ4 asks what retrieval, generation, and evaluation failures help explain disagreements. The final question requires examining actual evidence and reviewer judgments, because a lexical mismatch cannot automatically establish factual error.

The contribution is a reproducible empirical comparison rather than a new general-purpose retrieval algorithm. It includes a fixed dataset and corpus, a training-only logistic relevance model, 1,500 paired-condition responses, an integrity audit, question-level bootstrap intervals, and a systematic 24-question qualitative review. The central result is conditional: semantic retrieval and reranking improved observed means over BM25, but reranker gains depended on K, and additional context did not consistently help.

# 2. Related Work

Open-domain question answering separates finding evidence from using it. DrQA combined Wikipedia retrieval with a learned reader, showing why document search and reading comprehension must be assessed together [@drqa]. Original RAG combined parametric generation with a retrieved nonparametric memory and jointly fine-tuned components [@rag]. REALM incorporated retrieval into language-model pretraining [@realm], while latent retrieval research learned retriever-reader behavior from answer supervision [@orqa]. These architectures motivate evidence access, but they differ from the present experiment: its generator is a fixed API model, and only a lightweight relevance classifier is trained locally.

Sparse and dense retrieval encode different matching assumptions. The probabilistic relevance framework explains BM25's term-based scoring [@bm25]. Dense Passage Retrieval learns dual encoders for questions and passages [@dpr], whereas Sentence-BERT constructs reusable sentence representations that support cosine comparisons [@sbert]. The current semantic method uses a pretrained Sentence Transformer without question-answer fine-tuning. This distinction prevents attributing results to supervised dense-retriever training that was not performed. Research comparing sparse, dense, and attentional representations also identifies tradeoffs between lexical precision, fixed-vector capacity, and richer interactions [@luan].

Reranking revisits a candidate set rather than searching an unrestricted collection from scratch. BERT passage reranking uses learned query-passage interactions [@bert_rerank]; ColBERT retains token-level late interaction while permitting document-side precomputation [@colbert]. The present logistic model is intentionally simpler: four scalar features replace contextual pair encoding. Scikit-learn supplies consistent estimator interfaces for this supervised pipeline [@sklearn_paper]. Its transparency supports inspecting coefficient signs, although correlated features and class weighting prevent causal feature-importance interpretations.

These alternatives also expose distinct training and inference questions. A supervised dual encoder changes the representation space, while a reranker changes the ordering of already available candidates. Late interaction preserves finer matching information than a single passage vector [@dpr; @colbert; @luan]. Comparing their published headline scores would therefore mix architecture, supervision, corpus and reader changes. The current experiment instead holds those external conditions fixed where practical and asks whether a small learned combination improves over its own candidate-generating baselines. Because raw BM25 scores and cosine similarities use different scales, feature standardization is part of the implemented classifier rather than evidence that the underlying scores have identical meanings. This narrower comparison contributes an inspectable local result without claiming to reproduce the larger studies' training regimes.

Benchmark design determines what a retrieval comparison means. Natural Questions contains real search queries paired with Wikipedia annotations [@nq]. SQuAD established widely used span-oriented Exact Match and token-overlap evaluation [@squad], while its unanswerable extension emphasized recognizing when evidence does not support an answer [@squad2]. KILT evaluates provenance as well as downstream predictions against a shared Wikipedia snapshot [@kilt]. BEIR demonstrates that relative retrieval performance can change across heterogeneous tasks and domains [@beir]. Consequently, results on a selected answer-bearing corpus should not be presented as benchmark-wide or cross-domain superiority.

Retrieved-context effects are similarly architecture dependent. Fusion-in-Decoder showed that a trained generative reader could benefit from additional passages [@fid]. In contrast, long-context studies found sensitivity to evidence position [@lost_middle], and experiments with irrelevant arithmetic context demonstrated distractibility [@distracted]. These results justify examining K and evidence composition; they do not establish that any decrease here was caused by distraction. The current experiment changes passage depth but does not manipulate context positions or isolate individual distractors.

Evaluation research further separates answer overlap from evidence-supported factual precision. FActScore assesses atomic factual support in long-form generation [@factscore]; that metric was not implemented here. Instead, saved lexical scores are supplemented by explicit human review. Statistical comparisons likewise require attention to the evaluation unit. Bootstrap work in machine translation resamples matched examples [@koehn], and guidance for NLP significance testing stresses matching methods to tasks and experimental designs [@dror]. This study follows that principle by resampling question IDs, preserving condition pairing, and reporting pointwise uncertainty rather than treating 1,500 answers as independent questions.

# 3. Methodology

## 3.1 Dataset and preprocessing

The implementation loads the `LLukas22/nq-simplified` repackaging through Hugging Face Datasets [@nq_mirror; @datasets], pinned to revision `d0b4aa2c6b142fda852d4f4f371c1134560b69fa`. This derivative supplies question, context, and answer-text/character-offset fields; it is not the original Natural Questions evaluation protocol. Source training and test partitions were pooled in stable order. The preparation manifest records 106,921 rows and 65 duplicate questions, leaving 106,856 usable unique questions.

Questions, contexts, and answer annotations are validated, including exact answer-offset correspondence. Deduplication uses Unicode NFKC normalization, case folding, and word-token/whitespace normalization while preserving original question text. Python's seeded random sampling selects 700 questions from sorted normalized keys and partitions them into 500 training, 50 validation, and 150 test questions. These custom question splits remain disjoint. SHA-256 identifiers connect each question to its retained source context. Exact context deduplication yields 696 passages shared by all methods. Because the corpus is assembled from selected sources across the custom splits, test source texts are intentionally searchable; test question labels are withheld from model fitting. This is a controlled answer-bearing prototype, not an unbiased Wikipedia search collection.

## 3.2 Retrieval approaches

BM25 uses `rank_bm25.BM25Okapi` [@rank_bm25], lowercasing and extracting Unicode word tokens without stemming or stopword removal. Verified package defaults are k1=1.5, b=0.75, and epsilon=0.25; equal scores retain passage-ID order. Semantic retrieval uses `sentence-transformers/all-MiniLM-L6-v2`, a 384-dimensional encoder whose default input limit is 256 wordpieces [@minilm_card]. The saved model revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Unit-normalized float32 vectors are indexed in CPU FAISS `IndexFlatIP`; normalized inner products equal cosine similarity [@faiss_paper; @faiss_metrics]. Corpus vectors and passage IDs retain identical row order. Embedding truncation affects representation, while full original passages remain available to generation.

## 3.3 Trained logistic regression reranker

Candidates are the deduplicated union of BM25 top-10 and semantic top-10. Each pair receives BM25 score, semantic cosine similarity, the fraction of unique question tokens occurring in the passage, and full-passage word-token count. A pair is positive when its passage ID equals the question's annotated source; other candidates are labeled zero, meaning unannotated rather than necessarily irrelevant.

Training contains 8,794 pairs: 500 positives and 8,294 negatives. Twelve otherwise-missing training positives were inserted; no positive insertion occurred during validation or test retrieval. The pipeline fits StandardScaler on training pairs and transforms later pairs with those stored statistics [@scaler_docs]. LogisticRegression uses balanced class weights, L2 regularization with the default lbfgs solver, `max_iter=1000`, and `random_state=42` [@logistic_docs]. C values 0.01, 0.1, 1, and 10 were compared on validation MRR@10; C=0.01 was selected. Validation uses 853 natural candidate pairs from 50 questions. The selected training-fitted pipeline was saved without a train-plus-validation refit. Test candidates are sorted by predicted label-1 probability; these class-weighted scores are not calibrated factual-confidence estimates.

## 3.4 RAG answer generation

All conditions request `gpt-4.1-mini-2025-04-14` [@openai_model], temperature 0, a 64-output-token limit, and seed 42. Prompt version `nq-short-answer-v1` uses the same concise short-answer system instruction. RAG user content contains the question and ordered, numbered sources with passage IDs and complete text; it asks for an evidence-based answer or “Insufficient evidence.” Baseline questions omit documents and permit abstention. Reference answers and source labels are not model inputs. A five-question validation pilot preceded the frozen 150-question test. Usage, settings, retrieved IDs, prompts, latency, retries, and responses were saved; successful calls were cached. A single generation was obtained for each of ten conditions per test question.

## 3.5 Evaluation metrics

Saved Exact Match applies lowercase, ASCII-punctuation removal, English-article removal, and whitespace normalization, then maximizes equality across accepted references. Token F1 maximizes `2PR/(P+R)` across references, where precision and recall use multiset token overlap, following the declared SQuAD-style policy [@squad]. Empty or recognized abstained answers score zero. Hit@K is one if the single annotated source occurs within rank K, otherwise zero. MRR@10 averages reciprocal annotated-source rank, with zero when absent from top-10. These measure annotated-source recovery, not exhaustive relevance recall. Actual delivered-K source presence is recorded separately. Efficiency uses provider-reported input/output tokens and retrieval-plus-generation client milliseconds. Setup and deliberate pacing are excluded; retries/backoff and network effects remain. Monetary pricing was not configured.

## 3.6 Statistical analysis and integrity

Every comparison pairs the same 150 question IDs. For score difference `d(q)=score_A(q)-score_B(q)`, 10,000 seed-42 bootstrap samples resample 150 question IDs with replacement; the same index matrix is reused across comparisons. Percentile 2.5/97.5 limits describe mean paired differences. EM effects are percentage points; F1 effects are fractions. Intervals are pointwise, not multiplicity-adjusted, and do not remove selection bias. Question exchangeability is assumed; dependence from shared contexts is not separately modeled. The integrity audit reported 47 passing checks, eight warnings, zero failures, and two unverified checks. Frozen hashes, schemas, rankings, source mappings, score agreement, and train-only provenance evidence were inspected; no results were repaired for paper production.

# 4. Experimental Results

## 4.1 Overall answer accuracy

Table 1 reports all conditions; Figure 1 visualizes Exact Match. The LLM-only baseline achieved {{llm_only_0_mean_em}}% EM and {{llm_only_0_mean_f1}} F1. Every RAG condition had higher observed means. Across the nine RAG-versus-baseline comparisons, both pointwise EM and F1 intervals excluded zero in the positive direction. For reranker K=3, the EM increase was {{pair5_delta_em_pp}} percentage points, with interval [{{pair5_em_pp_ci_low}}, {{pair5_em_pp_ci_high}}]; the F1 increase was {{pair5_delta_f1}} [{{pair5_f1_ci_low}}, {{pair5_f1_ci_high}}]. This supports an improvement in annotated-answer agreement on these questions, not a universal factual-accuracy claim.

{{main_table}}

![Figure 1. Mean Exact Match for the nine RAG conditions; the dotted line represents the separate LLM-only baseline. Bars start at zero. The highest observed result is highlighted descriptively, without implying universal superiority.](../results/figures/figure_1_exact_match.pdf){width=95%}

## 4.2 Retrieval quality

Table 2 and Figure 2 compare shared top-10 rankings once per method. Semantic retrieval and reranking recovered annotated sources more often than BM25. Their MRR@10 values were {{semantic_mrr}} and {{reranker_mrr}}, versus {{bm25_mrr}} for BM25. The reranker's natural union contained the source for 148 of 150 questions, setting a candidate-availability ceiling before reranking. Source presence nevertheless did not ensure exact agreement: 639 of the 1,350 RAG records had a source in the delivered context but EM=0, spanning 83 distinct questions. Source misses occurred in 106 records across 28 questions. These overlapping configuration-level outcomes are not independent question counts.

{{retrieval_table}}

![Figure 2. Annotated-source Hit@K and MRR@10 from saved full rankings. Each method appears once because rankings are shared across generation depths. These measures do not exhaust all relevant evidence.](../results/figures/figure_3_retrieval_hit_at_k.pdf){width=95%}

## 4.3 Trained reranker comparison

Reranker K=3 achieved the highest observed EM, {{reranker_3_mean_em}}%, and F1, {{reranker_3_mean_f1}}. Against BM25 K=3, its EM effect was {{pair1_delta_em_pp}} points [{{pair1_em_pp_ci_low}}, {{pair1_em_pp_ci_high}}], with F1 effect {{pair1_delta_f1}} [{{pair1_f1_ci_low}}, {{pair1_f1_ci_high}}]. Against semantic K=3, the F1 effect was {{pair2_delta_f1}} [{{pair2_f1_ci_low}}, {{pair2_f1_ci_high}}]. Table 3 and Figure 3 show why that observation should not be generalized across K. At K=5 and K=10, reranker and semantic EM means were equal; their small F1 differences had intervals spanning zero. K=3 question-level F1 wins/ties/losses were 30/110/10 against BM25 and 12/132/6 against semantic, preserving unfavorable cases.

{{paired_table}}

![Figure 3. Selected paired mean F1 differences over the same 150 questions. Horizontal segments are saved percentile 95% bootstrap intervals from 10,000 resamples. Zero denotes no mean difference; intervals are pointwise and not adjusted for multiple comparisons.](../results/figures/figure_5_paired_f1_differences.pdf){width=100%}

## 4.4 Influence of K

Figure 4 shows nonmonotonic F1. BM25 decreased from K=3 to K=5, then increased at K=10; semantic retrieval peaked at K=5, and reranking peaked at K=3. Reranker K=10 minus K=3 changed F1 by {{reranker_k_change_delta_f1}}, with interval [{{reranker_k_change_f1_ci_low}}, {{reranker_k_change_f1_ci_high}}]. Semantic K=5 minus K=3 changed F1 by {{semantic_k_change_delta_f1}} [{{semantic_k_change_f1_ci_low}}, {{semantic_k_change_f1_ci_high}}]. Neither interval establishes a reliable directional population effect. Across within-method contrasts, some questions improved while others degraded, including cases where the annotated source remained available.

![Figure 4. Mean token F1 at K=3, 5, and 10, with the LLM-only baseline shown separately. The axis spans 0–1. These are descriptive means; paired-difference intervals are not misused as individual-mean error bars.](../results/figures/figure_2_f1_vs_k.pdf){width=95%}

## 4.5 Efficiency and latency

Figure 5 relates quality to input tokens. Reranker input usage rose from {{reranker_3_avg_input_tokens}} at K=3 to {{reranker_10_avg_input_tokens}} at K=10 while its mean F1 declined. All within-method K increases consumed more input tokens. Table 1 also reports short but nonidentical output-token means. Mean recorded total client latency ranged from approximately 581 to 1,049 ms. BM25 K=5's high mean and nonmonotonic times should be interpreted alongside two recovered API errors and seven unusual wall-clock gaps. Outliers were retained. Recorded retrieval means were approximately 4.5 ms for BM25, 27.5 ms for semantic, and 49.3 ms for reranking, the last including candidate retrieval. No dollar estimate or strong intrinsic-speed ranking follows from these noisy client measurements.

![Figure 5. Mean F1 versus actual mean input tokens for all ten configurations. Colors identify methods and labels distinguish K. Input tokens are usage units, not estimated monetary costs. More context did not consistently yield greater lexical answer quality.](../results/figures/figure_4_f1_vs_input_tokens.pdf){width=95%}

# 5. Qualitative Failure Analysis

The automated taxonomy separates source misses, source-present lexical failures, method advantages/regressions, higher-K changes, and baseline disagreements. Seed-42 selection combined three suggested starting cases, category coverage, possible formatting mismatches, and randomized fill to 24 unique questions. Every category had at least two eligible selected questions. Categories overlap and the sample is purposeful, not representative. All 24 primary comparisons were reviewed: 18 A responses were labeled correct, five uncertain, and one partially correct. Reviewer cause labels included nine lexical mismatches and six ambiguous references. These judgments do not estimate factual accuracy over 150 questions or 1,500 responses.

Table 4 summarizes six primary comparisons. For the Friends ending, semantic K=3 returned “Ross Geller,” while BM25 abstained. Semantic and reranking retrieved the finale source first; BM25 omitted it. The reviewer labeled A correct and supported. The observed advantage is associated with source availability, although it does not isolate the causal contribution of semantic matching or rule out useful alternative evidence.

For Bull Run, BM25 returned the accepted “Confederate forces,” while reranker K=3 produced a longer sentence containing that answer. Both supplied the annotated source first. The reviewer labeled the reranked answer correct and supported, with a lexical mismatch. Its EM=0 and F1=0.3077 therefore do not demonstrate a factual regression. In contrast, the broad-base population-pyramid response was reviewed as partially correct: the reviewer noted supported core content plus awkward additional wording. Its lower score is retained rather than explained away as harmless formatting.

The Witcher comparison illustrates ambiguity without a source-ranking improvement. Both K=5 prompts shared their first four passages, including the source at rank 1. That source distinguishes a 1986 short story from a 1990 collection. Semantic answered “1986”; reranking answered the accepted “1990.” The reviewer labeled A correct but categorized the cause as ambiguous reference. Different secondary context and interpretation remain plausible explanations, not established mechanisms.

Source availability also failed to prevent abstention. For “who sings the song sorry that i let you down,” BM25 retrieved the annotated passage naming NF but abstained. The reviewer marked factual correctness uncertain, evidence support supported, and cause abstention; lyric-versus-title wording was a possible explanation. This is not automatically a confirmed factually wrong assertion.

Finally, Hallmark channel answers demonstrate higher-K lexical degradation with retained evidence. BM25 K=3 returned “565 (HD),” while K=5 added “DirecTV channel”; the reference was “565.” F1 decreased from 0.6667 to 0.4000, although the reviewer labeled primary A correct and supported. Additional passages accompanied the change, but distraction was not experimentally isolated. Together, these cases distinguish retrieved-source access, interpretation, response formatting, and factual judgment.

{{case_table}}

# 6. Discussion

RQ1 is answered positively within this prototype: every RAG configuration improved mean lexical agreement over the no-retrieval baseline, and the paired intervals support positive sample-level differences. That result is narrower than proving that every retrieved response was factually better. RAG prompts also explicitly constrained generation to supplied evidence, whereas the baseline used parametric knowledge; the comparison concerns these complete conditions rather than an isolated mathematical retrieval effect.

For RQ2, dense retrieval's stronger annotated-source recovery is consistent with matching meaning across wording differences [@dpr; @sbert]. The lightweight reranker improved observed K=3 ordering and downstream quality, but little separated it from semantic retrieval at larger K. Its standardized semantic-similarity coefficient was positive and larger than the other coefficients; word overlap was negative conditional on the remaining features. Correlation, candidate selection and class weighting make these associations unsuitable as causal importance estimates. Without feature ablations, the separate contribution of each feature is unmeasured.

For example, reranker K=3 included the annotated source in 144 prompts but produced 73 exact matches. That difference is a lexical evaluation gap, not evidence of 71 factual errors. Additional ranking supervision cannot by itself correct reference wording. Candidate-pool coverage, delivered evidence availability, and generated-answer agreement should therefore remain separate diagnostics rather than interchangeable measures of one system property.

RQ3 reveals a practical quality-efficiency tradeoff: input consumption increased consistently, while accuracy did not. Trained generative architectures can benefit from aggregating more passages [@fid], but that does not guarantee the same behavior for this fixed prompted model. Context position, unrelated material and ambiguity are candidate explanations [@lost_middle; @distracted], not demonstrated causes here. Selecting the best observed test K is descriptive reporting, not a new tuning step or a validated deployment policy.

RQ4 identifies evaluation design as part of the explanation. Source recovery, lexical span agreement and reviewer-assessed correctness can disagree. The reviewed Muslim-population example is especially cautionary: a table's share of the world's Muslim population was used as the reference for a differently worded population-percentage question. The reviewer retained uncertainty despite lexical agreement. The pipeline preserved that annotation rather than modifying the benchmark after observing test results. Lightweight RAG design should therefore report candidate coverage, downstream scores, costs in observable units, and evidence-backed error analysis together, while reserving comprehensive factuality assessment for a separate evaluation [@factscore].

# 7. Limitations

The corpus contains 696 selected answer-bearing contexts, not unrestricted Wikipedia. It makes relevant sources available by construction and provides relatively few distractors. Only 150 test questions were evaluated, and the custom split was created from pooled source partitions. Earlier BM25-only test retrieval inspection means the test set was not historically untouched, even though final generation settings and classifier selection were frozen before the final evaluation.

One annotated passage per question does not exhaust relevance. Negative reranker labels can include useful evidence, and a source miss need not mean total evidence failure. Exact Match and F1 measure lexical agreement, not independently verified factual correctness. Some references are incomplete, ambiguous or time sensitive. The single reviewer examined a deliberately selected, overlapping sample; there is no inter-rater reliability estimate or representative factual-accuracy measurement.

Only one generation snapshot, embedding model and controlled corpus were evaluated, with one response per condition rather than repeated trials. Twenty-six provider backend fingerprints appeared despite fixed requested settings. Recorded timing includes client/network and retry effects; no pure inference benchmark or configured monetary-price calculation was available. Pointwise bootstrap intervals were not multiplicity-adjusted, and question resampling does not separately account for shared-context dependence or generation variability.

Training provenance is supported by code, hashes, scaler statistics and saved metadata, not independently signed historical fit logs. Local response records do not independently attest provider billing or internal execution. The audit therefore establishes consistency with documented limitations, not absolute correctness. No neural reranker, factual-verification model, answer-revision mechanism, context-position intervention, feature ablation or cross-corpus generalization experiment was conducted.

# 8. Conclusion

This study examined how retrieval strategy and passage count affect question answering when the corpus, generator and evaluation questions remain fixed. All nine RAG conditions exceeded the LLM-only baseline on mean Exact Match and token F1. Semantic retrieval improved observed means over BM25, while a training-only logistic relevance reranker achieved the highest observed result at K=3: {{reranker_3_mean_em}}% Exact Match and {{reranker_3_mean_f1}} F1. Paired uncertainty supports its K=3 differences on this sample, but not a consistent advantage over semantic retrieval at K=5 or K=10.

Increasing K raised input-token consumption without monotonic answer-quality gains. Source recovery and downstream correctness therefore require separate evaluation. The reviewed cases further showed why lexical failures can reflect wording, reference ambiguity, abstention or partially correct elaboration rather than a uniform factual-error mechanism. These observations support transparent reporting of both favorable and unfavorable outcomes.

Future work could extend the corpus with independent distractors, evaluate additional embeddings and cross-encoder rerankers, obtain more comprehensive relevance judgments, and measure factuality through semantic or multi-reviewer evaluation. Repeated generation trials would help separate context effects from generation variability. These are proposed extensions, not completed experiments. The present contribution is a controlled, auditable comparison with explicit scope and uncertainty, rather than a claim that one retrieval strategy universally produces more accurate RAG systems.
