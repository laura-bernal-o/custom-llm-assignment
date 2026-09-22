# My Custom LLM Experiment

Class 4 assignment — training [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
(word-token classroom build) from scratch, twice, and evaluating it honestly.
Built from the [starter repo](https://github.com/pepealonso95/custom-llm).

Grading uses deliverable quality **4 points**, testing & evaluation **3 points**,
and working result **3 points**. My model's eval percentage is *not* my grade —
what follows is complete, honest evidence for both experiments, including the
parts that didn't work.

## My choices and prediction

- **Corpus:** started with the supplied synthetic classroom corpus
  (`CORPUS = "classroom"`), then ran a second experiment with two files added to
  `corpus/`: [`negation.txt`](corpus/negation.txt) and
  [`opposites.txt`](corpus/opposites.txt) — **591 new unique passages**,
  targeting the eval suite's `negation` and `opposites` extension categories.
  Both files are text I generated myself from simple sentence templates (no
  external/copyrighted source, no PDFs), so there are no permission concerns.
  I deliberately used **different names, objects, and adjective pairs** than the
  eval suite's fixtures (the eval uses `hot/cold`, `empty/full`, `noisy/quiet`,
  `ava/tea/milk`, `box/red/blue`, `door/open/closed` — my corpus uses
  `big/small`, `wet/dry`, `sharp/dull`, `maya/leo/...` did-not-buy patterns with
  different objects, etc.) so I teach the *grammatical pattern*, not the
  specific test answers. See [`corpus_manifest.json` (starter run)](llm_runs/20260922T173600_827803Z/corpus_manifest.json)
  and [`corpus_manifest.json` (expanded run)](llm_runs/20260922T173848_430218Z/corpus_manifest.json)
  for exact filenames, hashes, and passage counts.
- **Training steps:** 10 steps first (setup check — confirmed the whole pipeline
  ran end to end), then **3,000 steps** for both real experiments, matching the
  assignment's suggested starting budget. One step updates weights on a batch
  of 32 documents, not the whole corpus, so 3,000 steps means roughly 96,000
  document-draws with repetition against ~4,600–5,200 unique passages.
- **Learning rate: 0.001**, with built-in warmup + cosine decay. This is a
  standard conservative AdamW starting point for a model this size: large
  enough to make real progress in a few thousand steps, small enough (with
  warmup) to avoid loss spikes early in training. A much higher rate risks the
  loss oscillating or diverging before warmup finishes; a much lower rate would
  mean 3,000 steps barely moves the loss at all.

**My full written prediction, made before training, is in the notebook itself**
(section 1 of [`custom_llm.py`](custom_llm.py), reproduced in both executed
notebooks): I expected training/validation loss to fall together and level off
well below the untrained baseline (near ln(vocab size)), expected `starter_patterns`
eval cases to improve the most since they're literally the trained sentence
frames, expected `starter_transfer` to improve less, and expected the 24
`extend_corpus` cases to **not** improve from more starter-corpus training,
since that vocabulary simply isn't there. All of that held. What I did *not*
fully predict: that adding negation/opposites *pattern* training with
deliberately different words would still leave the eval's specific `negation`/
`opposites` cases at 0/3 each — see [Fixed language evals](#my-fixed-language-evals) below.

## My run

- **Executed notebooks:** [`custom_llm.ipynb`](custom_llm.ipynb) (starter-corpus
  experiment) and [`custom_llm_corpus_extension.ipynb`](custom_llm_corpus_extension.ipynb)
  (expanded-corpus experiment). Both ran `Run All` top to bottom via
  `jupyter nbconvert --execute`, no interruptions, no errors.
- **Source of truth:** [`custom_llm.py`](custom_llm.py) (a jupytext-style
  script — both notebooks are built from it with `python build_notebook.py`,
  then executed; the only difference between the two runs is whether
  `corpus/negation.txt` and `corpus/opposites.txt` exist at run time).
- **Hardware:** Apple Silicon Mac, macOS 14, CPU only (`DEVICE = "cpu"`),
  PyTorch 2.14.0, Python 3.12.0.
- **Elapsed time:** starter run **20.1 s** for 3,000 steps
  ([`training_summary.json`](llm_runs/20260922T173600_827803Z/training_summary.json));
  expanded run **21.0 s** for 3,000 steps
  ([`training_summary.json`](llm_runs/20260922T173848_430218Z/training_summary.json)).
  Neither run was interrupted.
- **Parameter count:** 111,872 params (starter, vocab 136) and 119,424 params
  (expanded, vocab 254) — the 2-block/4-head/64-dim/48-token-context config from
  the assignment; the only size difference is the embedding/output table growing
  with vocabulary.

| | Starter corpus | Expanded corpus |
|---|---:|---:|
| Vocabulary size | 136 | 254 |
| Training unknown-token rate | 0.00% | 0.00% |
| Held-out unknown-token rate | 0.00% | 0.00% |
| Train / validation documents | 4,132 / 460 | 4,664 / 519 |
| Reserved eval passages excluded before split | 160 | 160 |

Both [vocabulary_report.json (starter)](llm_runs/20260922T173600_827803Z/vocabulary_report.json)
and [vocabulary_report.json (expanded)](llm_runs/20260922T173848_430218Z/vocabulary_report.json)
show 0% unknown-token rates because the 509-type cap was never hit (133/251
distinct training word types respectively) — every training word type was
retained, so nothing was lost to the vocabulary cap in either run. The split is
by deduplicated **passage**, not source file, so this tests new combinations of
already-seen sentence templates, not generalization to unseen documents — the
[`split.json`](llm_runs/20260922T173600_827803Z/split.json) files record the exact
train/validation passage lists.

## My evidence

### Loss and samples

![Starter training curve](llm_runs/20260922T173600_827803Z/training_curves.svg)
![Expanded training curve](llm_runs/20260922T173848_430218Z/training_curves.svg)

Fixed evaluation panels: **20 training documents, 20 validation documents** in
both runs (`evaluation_panel_size` in [config.json](llm_runs/20260922T173600_827803Z/config.json)).

| Step | Starter — train loss | Starter — val loss | Expanded — train loss | Expanded — val loss |
|---:|---:|---:|---:|---:|
| 0 | 4.9263 | 4.9275 | 5.5819 | 5.5528 |
| 1500 | 0.6821 | 0.7182 | 0.7783 | 0.7569 |
| 3000 | 0.6783 | 0.7061 | 0.7394 | 0.7241 |

Full data: [history.json (starter)](llm_runs/20260922T173600_827803Z/history.json),
[history.json (expanded)](llm_runs/20260922T173848_430218Z/history.json),
and the raw [training.csv (starter)](llm_runs/20260922T173600_827803Z/training.csv) /
[training.csv (expanded)](llm_runs/20260922T173848_430218Z/training.csv).
The expanded run's untrained loss starts higher (5.58 vs 4.93) simply because
its vocabulary is larger (254 vs 136 types), so a uniform random guess has a
lower baseline probability — this is expected and not comparable across runs
as an absolute number.

**Untrained → halfway (1,500 steps) → final (3,000 steps) samples**, same
generation settings throughout (temperature 0.8, seed 2026, no weight updates
during generation):

| Stage | Starter corpus | Expanded corpus |
|---|---|---|
| Untrained | `pear professor bond doctor course harvest team physician journey checking buyer delivery traffic report the lecturer item offering and system <UNK> taste recommended mentioned bus question customer at mortgage nurse in instructor` | `ticket cap taste , bike kite` |
| Halfway (1500) | `our school has a question about the new educator and lesson .` | `the report about the pear explains the juice in detail .` |
| Final (3000) | `the report about the nurse explains the health in detail .` / `the consumer compared the offering after checking the price .` | `the report about the consumer explains the service in detail .` / `we learned about the important therapist during a discussion of patient .` |

Full sample files (4 samples per checkpoint, including any empty/garbled ones):
[starter `samples/`](llm_runs/20260922T173600_827803Z/samples/),
[expanded `samples/`](llm_runs/20260922T173848_430218Z/samples/).

**What changed:** the untrained model outputs unstructured word salad (correct
*token types* for the corpus, no structure). By 1,500 steps both models already
produce grammatical, on-template sentences. Training loss and generated-text
quality both plateau by 3,000 steps — more steps on this corpus would likely
not help much further, since the corpus is a small set of repeated templates.

### Tokens, embeddings, gradient, and one weight update

Traced word: **`customer`** (in both runs, since it's a shared starter-corpus word).

| | Starter (ID 28, vocab 136) | Expanded (ID 53, vocab 254) |
|---|---|---|
| Initial vector (first 8 of 64 numbers) | `[-0.058, -0.005, 0.043, 0.019, 0.016, -0.029, 0.026, 0.000]` | `[0.002, 0.001, -0.005, -0.052, 0.009, -0.021, 0.041, -0.039]` |
| Trained vector (first 8 of 64 numbers) | `[0.037, -0.018, 0.133, 0.106, 0.063, 0.019, 0.152, 0.093]` | `[0.079, 0.034, 0.007, 0.008, -0.136, 0.089, 0.021, -0.066]` |

Full 64-number vectors, before and after: [`inspection.json` (starter)](llm_runs/20260922T173600_827803Z/inspection.json),
[`inspection.json` (expanded)](llm_runs/20260922T173848_430218Z/inspection.json).
Token IDs are arbitrary row indices assigned by sorted frequency — they carry no
meaning by themselves; the 64 numbers in that row *are* the learned meaning, and
they moved substantially from their random initialization.

**First recorded gradient and parameter update** (step 0, coordinate 0 of
`customer`'s embedding, starter run):

```
before    = -0.057591915130615234
gradient  =  0.0006925869965925813
lr (step) =  1e-05                    # warmup: lr ramps up from ~0 toward 0.001
after     = -0.057601906359195710
```

`after - before = -0.0000099912...`, i.e. almost exactly `-1 × lr`
(`-0.99912 × lr`), **not** `-lr * gradient` (which would be `-6.9e-9`, about
1,400× smaller). This is a real, checkable illustration of why the training
cell's comment says AdamW "is not simply learning-rate times gradient": on
its very first step, Adam's bias-corrected first/second-moment estimates both
reduce to the raw gradient itself (`m̂ = g`, `v̂ = g²`), so the update becomes
`-lr × g/(|g|+ε) ≈ -lr × sign(gradient)` — the optimizer moves the parameter
by (approximately) one learning-rate step in the direction that reduces loss,
regardless of how large or small the gradient itself was. Later steps, once
the running averages have accumulated history from many gradients, behave
differently. Warmup also kept this very first step's learning rate tiny —
`1e-05`, not the full `0.001`.

**Next-token probability comparison** for the prefix `"the customer"` (starter run):

| | Before training (top 5) | After training (top 5) |
|---|---|---|
| Predictions | `customer` 1.6%, `bus` 1.1%, `educator` 1.0%, `us` 1.0%, `application` 1.0% | `reviewed` 17.8%, `recommended` 17.1%, `ordered` 16.9%, `selected` 16.3%, `compared` 16.0% |

Before training the distribution is nearly flat (all candidates within a point
of each other — essentially noise from random initialization). After training
the model heavily favors the verbs that actually follow "the customer" in the
training frames (`the customer ordered/reviewed/compared/returned/recommended/
selected the product ...`) — a direct, inspectable readout of what the network
learned. Full distributions: [`inspection.json`](llm_runs/20260922T173600_827803Z/inspection.json).

### Temperature comparison

Same starting conditions (seed 2026, no weight updates), three temperatures,
starter run: [`temperature_comparison.json`](llm_runs/20260922T173600_827803Z/temperature_comparison.json)
(also [expanded](llm_runs/20260922T173848_430218Z/temperature_comparison.json)).
At 0.3 (sharper distribution) the four samples are nearly identical, safe,
high-probability continuations. At 0.8 there's more variety while staying
grammatical. At 1.2 the expanded-corpus run produced `noah wanted the cap .` —
a *novel* sentence combining a negation-corpus name (`noah`) with a
negation-corpus object (`cap`) in a pattern from the classroom corpus, which is
a nice concrete sign that the added vocabulary is actually blended into the
network's generation, even though (as below) it doesn't help the specific
eval cases. No weights changed between temperatures — temperature only rescales
the logits before sampling; it's a decoding-time choice, not a training one.

## My fixed language evals

Suite: [`evals/language_evals.json`](evals/language_evals.json) (unchanged, 48
cases). Runner: [`run_evals.py`](run_evals.py). All four required result sets:

- [`results/starter-untrained/`](results/starter-untrained/) · [`llm_runs/.../language_evals/untrained/`](llm_runs/20260922T173600_827803Z/language_evals/untrained/)
- [`results/starter-final/`](results/starter-final/) · [`llm_runs/.../language_evals/final/`](llm_runs/20260922T173600_827803Z/language_evals/final/)
- [`results/expanded-untrained/`](results/expanded-untrained/) · [`llm_runs/.../language_evals/untrained/`](llm_runs/20260922T173848_430218Z/language_evals/untrained/)
- [`results/expanded-final/`](results/expanded-final/) · [`llm_runs/.../language_evals/final/`](llm_runs/20260922T173848_430218Z/language_evals/final/)

The `results/*` folders were produced by rerunning `run_evals.py` directly
against the saved `model.pt` / `model_untrained.pt` files (not the in-notebook
run) — every number below matches the notebook's own eval cells exactly,
confirming the saved checkpoints reproduce the recorded scores.

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable cases | Full results |
|---|---|---:|---:|---:|---|
| Starter corpus | Untrained | 9 | 24 | 37.5% | [eval_summary.json](results/starter-untrained/eval_summary.json) |
| Starter corpus | Trained | 20 | 24 | 83.3% | [eval_summary.json](results/starter-final/eval_summary.json) |
| Expanded corpus | Untrained | 6 | 24 | 25.0% | [eval_summary.json](results/expanded-untrained/eval_summary.json) |
| Expanded corpus | Trained | **24** | 24 | **100%** | [eval_summary.json](results/expanded-final/eval_summary.json) |

Coverage (scorable/48) is **50% in every stage of both experiments** — the
`extend_corpus` group's 24 cases are entirely out-of-vocabulary in both runs,
so "scorable" never moves; only accuracy *among the 24 scorable cases* moves.

### By category (trained models)

| Category | Starter trained | Expanded trained |
|---|---:|---:|
| domain_context (8) | 8/8 | 8/8 |
| domain_place (8) | 8/8 | 8/8 |
| new_wording (8) | 4/8 | **8/8** |
| grammar (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| opposites (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| negation (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| reference (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| sequence (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| spatial_relations (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| everyday_knowledge (3) | 0/3 (unscorable) | 0/3 (unscorable) |
| categories_and_analogies (3) | 0/3 (unscorable) | 0/3 (unscorable) |

**What worked:** every `starter_patterns` case (the exact trained sentence
frames) reached 16/16 in *both* trained runs. `new_wording` (new phrasings of
starter vocabulary) went from 4/8 to a perfect **8/8** in the expanded run —
this is a real, interesting improvement, and my best guess for why is that
adding 591 more passages using the same core grammar (subject/object/negation
sentences) gave the model more exposure to sentence-final punctuation and
clause boundaries generally, which is exactly the kind of pattern
`new_wording` cases probe, even though those specific cases don't use
negation/opposites vocabulary.

**What did not work, and why:** all 6 of my target `negation`/`opposites`
cases remained **out-of-vocabulary in both experiments**, unimproved. I
checked the exact reason in the per-case results
([`results/expanded-final/eval_results.json`](results/expanded-final/eval_results.json)):
every one of those 6 cases needs a specific content word — `hot`, `empty`,
`noisy`, `box/red/blue`, `ava/tea/milk`, `door/open/closed` — that never
appears anywhere in my corpus, starter or extension, **by design**: the eval
guide explicitly says "never insert eval words directly into the vocabulary
just to make the tests scorable," and instructs teaching the pattern with
*different* wording. I did exactly that. The result is an honest, expected
outcome: **learning a grammatical pattern and having the specific vocabulary a
test happens to use are two separate things**, and this tiny word-level,
509-token-capped model has no way to generalize a pattern to a word it has
literally never seen a token for. This is not a bug — it's the intended
lesson (the eval suite's `extend_corpus` group was designed to require adding
new vocabulary + patterns together, and the assignment explicitly forbids
gaming coverage by copying test vocabulary in). Free-text evidence backs this
up: `the opposite of hot is` → generated `young .` (plausible antonym
*shape*, wrong word, because `hot` itself is `<UNK>`); my own in-corpus
prompt `the opposite of big is` → `short .` in the terminal chat below, which
*is* a real (if imperfect) antonym-style completion, using vocabulary that
*is* in the corpus.

### Leakage / separation checks

[`eval_separation.json` (starter)](llm_runs/20260922T173600_827803Z/eval_separation.json)
and [(expanded)](llm_runs/20260922T173848_430218Z/eval_separation.json) both
record **160 generated classroom passages excluded** before the train/validation
split and vocabulary build, covering the 16 `starter_patterns` case IDs whose
exact prefixes appear in the generated classroom sentences. I additionally
ran `reject_eval_leakage()` by hand against both new corpus files before
training and confirmed zero matches (the notebook's own import-time check
would otherwise have raised and stopped the run). I wrote `corpus/negation.txt`
and `corpus/opposites.txt` using names, objects, and adjective pairs chosen
specifically to avoid every content word in the suite's 6 negation/opposites
cases — see [My choices](#my-choices-and-prediction) above. These checks are
normalized-text matches, not semantic ones — they would not catch a rephrased
copy of a test item, only exact reserved prefixes, which is why I also
manually reviewed both corpus files against the suite before training.

## My chat interface

**Launch:**
```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python chat.py --model llm_runs/20260922T173848_430218Z/model.pt --transcript results/my-chat.json
```
Type a prompt and press enter; type `/quit` to exit. Each prompt starts a
fresh 48-token context — there is no conversation memory. This is the
**expanded-corpus, trained** model (3,000 steps, vocab 254).

Saved transcript (4 real interactions, exceeding the 3-minimum):
[`results/expanded-chat.json`](results/expanded-chat.json)

| You | Model |
|---|---|
| `the customer` | `selected the item after checking the price .` |
| `the opposite of big is` | `short .` |
| `maya did not want the soda . she wanted` | `the lesson .` *(unknown word: "she" — I never taught pronouns, only repeated names)* |
| `leo is not strong . he is` | `the important orange .` *(unknown word: "he" — same limitation)* |

**Limitation demonstrated:** the model never saw third-person pronouns
(`she`/`he`) in any corpus, starter or extension — my negation sentences always
repeat the subject's name (`maya did not want ... maya wanted ...`) instead of
using a pronoun. As soon as a prompt uses `she`/`he`, that token becomes
`<UNK>` and the completion falls apart, even though the surrounding
negation/opposites vocabulary is otherwise known. This is a concrete gap I'd
close in a next corpus iteration (see below). Section 10 of
[`custom_llm.py`](custom_llm.py) also drives a live in-notebook prompt/reply
cell — 3 separate prompts are run there in each experiment (visible directly
in both executed `.ipynb` files and each run's own
`chat_transcript.json`), independent of the 4-turn terminal transcript above.

### Bonus: web chat interface

Beyond the required terminal/notebook interfaces, [`webapp/`](webapp/) has a
small local web page for the same model. It's a thin wrapper, not a new
inference path: `webapp/server.py` is a Flask app that calls the *exact same*
`load_model` / `generate_reply` functions from [`run_evals.py`](run_evals.py)
that `chat.py` uses — no retraining, no new model logic, fresh 48-token
context per message, same as every other interface here.

**Launch:**
```sh
pip install -r webapp/requirements.txt
python webapp/server.py --model llm_runs/20260922T204815_706083Z/model.pt --port 5050
# then open http://localhost:5050
```

The page includes onboarding for a first-time visitor: a collapsed
step-by-step explainer, clickable example prompts (including ones that
deliberately trigger the unknown-word case), and a searchable browser over
every word the model actually knows (`/api/vocab`). When a prompt contains an
unknown word, the page shows an explanatory warning box *before* the reply
and visually marks that reply as unreliable, instead of letting it look like
a normal answer. A **temperature slider** (0.1–1.5, sent per-request to
`/api/chat`) lets you control sampling live — e.g. `the opposite of big is`
is the model's top prediction 29% of the time at the default 0.8 (it can also
land on `not`, `dry`, or rarer words), but at 0.2 it reliably returns `small`.
Each reply is labeled with the temperature that produced it. No weights ever
change; this only affects how the next-token distribution is sampled.

Its default model is a **third, exploratory run** (`llm_runs/20260922T204815_706083Z/`,
same 3,000 steps / lr 0.001 settings) — not one of the two graded experiments
above. It adds one more corpus file, [`corpus/people.txt`](corpus/people.txt)
(person/family words: girl, boy, mother, father, sister, brother, friend,
neighbor, etc., taught inside the same sentence patterns already learned),
purely so the chat interface has more to talk about. It's included here for
transparency, not as part of the required starter/expanded-corpus comparison.

| You | Model | Notice |
|---|---|---|
| `the customer` | `selected the item after checking the price .` | — |
| `the girl bought the` | `scooter .` | — |
| `the father is strong , not` | `weak .` | — |
| `maya did not want the soda . she wanted` | *(unreliable)* | Unknown word: `she` — pronouns were never taught |

Full interaction history: [`webapp/webapp_chat_log.json`](webapp/webapp_chat_log.json).

## What I learned

1. **Corpus and held-out data.** My corpus is ~4,600–5,200 short, highly
   repetitive template sentences across 8 business/service domains, plus (in
   the second run) negation and opposite-pair sentences. It can teach exactly
   the local word co-occurrence patterns it contains and nothing else — it
   cannot teach spatial relations, sequencing, references, or general facts,
   because those words and patterns are simply absent. I hold out 10% of
   deduplicated passages so I can measure loss on sentences the model never
   trained on directly — but because validation reuses the *same templates* as
   training, it tests recombination within known patterns, not generalization
   to new writing.
2. **Token vs. ID vs. vector vs. embedding.** A token is a unit of text (a
   word or punctuation mark here). An ID is just its row number in the
   vocabulary table — arbitrary, assigned by frequency rank, and meaningless
   on its own (`customer` is ID 28 in one run and ID 53 in the other, same
   word, same eventual behavior). The **vector** is the 64 actual numbers at
   that row. The **embedding table** (or "embedding") is the whole matrix —
   every token's vector, all learned together. Meaning lives entirely in the
   numbers, never in the ID.
3. **What makes this a neural network, and how it learns.** Two transformer
   blocks map token+position embeddings through causal self-attention and a
   feed-forward layer to output scores (logits) over the vocabulary. Loss
   (cross-entropy) measures how much probability the model put on the actual
   next token. `loss.backward()` computes the gradient of that loss with
   respect to *every* parameter (I saved one: `customer`'s first embedding
   coordinate had gradient `0.00069`, meaning "increasing this number very
   slightly would have increased the loss slightly, so decrease it"). AdamW
   then updates the parameter using that gradient plus momentum and
   per-parameter adaptive scaling — not just `lr * gradient` — which is why my
   step-0 update (`-0.0000099913`) is close to but not exactly `-1e-5 *
   0.00069`.
4. **Attention and context.** Attention lets each position mix in information
   from *earlier* tokens in the same sequence, weighted by a learned
   compatibility score between queries and keys; a triangular mask forces
   every position to ignore anything after it, which is what makes this a
   valid next-token predictor instead of a model that could "cheat" by
   peeking ahead.
5. **Probabilities, sampling, and temperature.** The final layer produces one
   score per vocabulary word; softmax turns those into a probability
   distribution; generation samples from that distribution one token at a
   time and feeds the result back in. Temperature divides the logits before
   softmax: lower temperature sharpens the distribution toward the single
   most likely word (safer, more repetitive), higher temperature flattens it
   (more variety, more mistakes). No gradients are computed and no weights
   change during generation at any temperature — it's purely a decoding-time
   choice, confirmed by `temperature_comparison.json` sharing the same
   `model_sha256` throughout.
6. **Did the evidence support my prediction?** Mostly yes: loss and samples
   improved exactly as predicted, `starter_patterns` reached 16/16 in both
   runs, and `extend_corpus` stayed unscorable from starter-only training. The
   one thing I underestimated: I expected adding negation/opposites *pattern*
   training to at least partially help the matching eval categories, but with
   deliberately non-overlapping vocabulary (which the assignment requires) it
   couldn't — a real, honest limitation, not a hidden success.

## One limitation and my next experiment

**Limitation:** the model has no pronouns (`she`, `he`, `it` as a referring
pronoun rather than the fixed idiom "it is ADJ") anywhere in its vocabulary,
because both my negation corpus and the starter corpus avoid them. This breaks
naturally-phrased negation prompts, as shown in the chat transcript above, and
is exactly the kind of narrow, inspectable gap this small a model makes easy
to find.

**Next experiment:** add a third corpus file introducing third-person
pronouns paired with the existing names (`maya did not want the soda . she
wanted the coffee .` — deliberately using a pronoun instead of repeating the
name), covering all 24 names already in `negation.txt` so the pattern is well
represented, and retrain. Prediction: `he`/`she` would join the retained
vocabulary (training-type count is well under the 509 cap, so nothing would be
evicted), the pronoun-using chat prompts above should stop showing "unknown
word," and I'd expect a further improvement in `new_wording`/`starter_transfer`-style
generalization tests, though I would still expect the 6 fixed `negation`/
`opposites` eval cases to stay unscorable, since their specific content words
(`hot`, `ava`, `box`, `door`, etc.) still wouldn't be taught — confirming that
vocabulary coverage for *those specific words* and pattern learning really are
separate levers.

## Reproduce and inspect

```sh
git clone <this repo>
cd custom-llm-assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter nbconvert ipykernel

# Starter-corpus experiment (corpus/ empty except README.md)
python build_notebook.py   # rebuilds custom_llm.ipynb from custom_llm.py, unexecuted
jupyter nbconvert --to notebook --execute --inplace custom_llm.ipynb

# Expanded-corpus experiment (with corpus/negation.txt + corpus/opposites.txt present)
jupyter nbconvert --to notebook --execute --inplace custom_llm.ipynb
mv custom_llm.ipynb custom_llm_corpus_extension.ipynb

# Rerun the fixed evals against any saved model
python run_evals.py --model llm_runs/<run>/model.pt --output results/my-final
python run_evals.py --model llm_runs/<run>/model_untrained.pt --stage untrained --output results/my-untrained

# Chat with a trained model
python chat.py --model llm_runs/<run>/model.pt --transcript results/my-chat.json
```

Each `Run All` creates a fresh timestamped folder under `llm_runs/`, so
re-running never overwrites the two runs already committed here:
`llm_runs/20260922T173600_827803Z/` (starter) and
`llm_runs/20260922T173848_430218Z/` (expanded). Open
[`embedding-viewer.html`](embedding-viewer.html) directly in a browser and use
**Open your checkpoint** to load either run's `checkpoint.json` for interactive
3D inspection of the token embeddings (PCA-projected; cosine neighbors use all
64 dimensions).

This repository's `.gitignore` was changed from the upstream template: the
template ignores `corpus/`, `llm_runs/`, and `results/` by default (to protect
students who add private files), but since every file here is my own original,
non-sensitive teaching text and the assignment requires linking these results
from a public repo, I deliberately track them.
