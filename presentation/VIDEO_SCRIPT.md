# Video Script: Aspect-Based Sentiment Analysis of Product Reviews

**Target length:** about 17–18 minutes (≈ 2,600 spoken words at a relaxed pace).
**Speakers:** Mrigank (slides 1–5) · Varshit (slides 6–11) · Jeshwanth (slides 12–17 + live demo)
**What you need open:** `presentation/ABSA_Final_Presentation.pptx` in slideshow mode, and `ABSA_Notebook.ipynb`
in Jupyter / VS Code (scrolled to section 9) for the demo.

How to use this script:
- Lines in *italics* are stage directions (what to click or show). Everything else is meant to be said out loud.
- You don't have to read it word for word. Practise once, then say it in your own way; it will sound more natural.
- The timings are cumulative, so you can check your pace while recording.
- Every number here matches the slides and the report.

---

## Part 1 · Mrigank (≈ 5 min)

### Slide 1 · Title (0:00 – 0:45)
*Slideshow on slide 1. Face camera if you are recording yourselves.*

Hi everyone. I'm Mrigank, and together with Varshit and Jeshwanth, I'll be presenting our project on aspect-based
sentiment analysis of product reviews.

In one sentence: we built a system that reads a review, finds the parts of the product the reviewer talks about, and
works out whether they liked each part or not. Then we compared old-school methods with modern Transformer models, and
we checked how well those models hold up when we move them to a different dataset.

On the right you can see our headline results. About 80 macro-F1 on laptop reviews, about 85 on restaurant reviews,
and about 85 F1 for finding the aspects themselves. We'll explain what all of that means as we go.

### Slide 2 · Recap: the task (0:45 – 2:00)
*Next slide.*

Let's start with the problem. Look at this sentence: "The laptop has an excellent display, but the battery life is
disappointing."

If you ask a normal sentiment classifier whether this review is positive or negative, it's stuck. It's both. Whatever
single label it picks, you lose the information you actually care about, which is that the display is good and the
battery is bad.

So we split the problem into two steps. The first step is **aspect term extraction**: find the words that name an
aspect, like "display" or "battery life". We treat it as tagging each word with B, I or O, where B means "an aspect
starts here", I means "the aspect continues", and O means "not an aspect".

The second step is **aspect sentiment classification**. We give the model the sentence and one aspect, and it tells
us positive, negative or neutral for that aspect.

And on the right is our evaluation protocol. Every model is trained on one dataset and tested on both, which gives us
four settings: two in-domain and two cross-dataset.

### Slide 3 · Data (2:00 – 3:15)
*Next slide.*

We used two public datasets. The first is SemEval-2014 Laptops, the standard benchmark for product reviews. The
second is MAMS, which is restaurant reviews, and it was built on purpose so that every single sentence has several
aspects with different sentiments.

You can see the sizes in the table. SemEval didn't come with a validation set, so we held out 10% of its training
sentences ourselves.

But the numbers at the bottom are the interesting part. Only 11% of laptop sentences mix positive and negative
opinions, while in MAMS it's 100%. MAMS sentences also have more aspects, 2.6 on average against 1.6. And look at the
red number: the overlap between the aspect vocabularies of the two datasets is less than one percent. Laptop
reviews talk about batteries and keyboards, restaurant reviews talk about food and waiters.

Keep that in mind, because it explains a lot of what you'll see in the cross-dataset results.

### Slide 4 · What we built (3:15 – 4:15)
*Next slide.*

Here's the pipeline, from left to right. We take the raw XML files, MAMS gets downloaded automatically, and we
preprocess everything in parallel on all CPU cores: parsing, tokenising and creating the BIO tags. From that we get two
views of the data, one for extraction and one for sentiment. Then we train the models, evaluate them in all four
settings, and analyse the results.

All of it runs from one command. Training happened on a laptop GPU, an RTX 4060 with 8 gigabytes. To make that fast we
kept the whole dataset in GPU memory and used mixed precision. The complete first round of experiments took about
17 minutes.

And every run tracks its own progress: training and validation loss and scores several times per epoch, with early
stopping on the validation set.

### Slide 5 · Round 1 models (4:15 – 5:00)
*Next slide.*

In the first round we compared three kinds of models under identical conditions.

On the left is the traditional baseline: TF-IDF features with logistic regression or an SVM. A plain bag of words
doesn't know which aspect we're asking about, so to make it fair we added features from a small window of words
around the aspect.

In the middle are the Transformers, BERT and RoBERTa. They read the sentence and the aspect together as a pair, and
classify from the special CLS token.

And on the right is the evaluation: same splits and same metrics for everybody, and the test set is touched only once
at the very end.

Now I'll hand over to Varshit for the results.

---

## Part 2 · Varshit (≈ 6 min)

### Slide 6 · Round 1 results: sentiment (5:00 – 6:15)
*Next slide.*

Thanks, Mrigank. So, how did round one go?

This table shows macro-F1 for sentiment classification. The first thing to notice is that the Transformers win
everywhere. On laptops, RoBERTa gets 77.5 against about 61 for TF-IDF. That's a gap of around 17 points. On MAMS the
gap is about 14 points.

The second thing is the cross-dataset columns, and they're lopsided. If you train on MAMS and test on laptops, RoBERTa
still gets 72, which isn't far from its in-domain score. But if you train on laptops and test on MAMS, it drops to 48.

Our explanation is this: laptop sentences mostly carry a single opinion, so a model trained on them learns to read
the general tone of the sentence. That shortcut completely fails on MAMS, where the aspects in a sentence always
disagree.

You can also see the same pattern in the heat maps underneath, for every model.

### Slide 7 · Round 1 results: extraction (6:15 – 7:15)
*Next slide.*

For aspect extraction, the story is simpler. Within a domain it works well: RoBERTa reaches about 86 F1 on laptop
aspects.

But across domains it falls apart. Look at laptop to MAMS: precision is 76%, but recall is only 31%. So whatever the
model finds is usually right, but it misses most restaurant aspects, simply because it has never seen words like
"sushi" or "waiter" being aspects.

The bottom table is the full pipeline, where an aspect only counts if we both find it and get its sentiment right.
That number is naturally lower, because every aspect we miss in step one is also a sentiment we can't get right.

### Slide 8 · Diagnosis (7:15 – 8:45)
*Next slide. Point at the curves.*

At this point the numbers were decent, but we wanted to know why they weren't better. So we looked carefully at the
training curves. This one is RoBERTa on MAMS. Blue is training, orange is validation.

We found four problems.

**One, over-fitting.** The training score goes all the way to 98, but validation stalls around 83. And the
validation loss actually starts going up after about one and a half epochs. The model is just memorising.

**Two, aspect localisation.** The classifier only looks at the CLS token, and nothing tells it where the aspect is in
the sentence. That's a real problem when a sentence has three aspects with three different labels.

**Three, the neutral class.** Neutral F1 on laptops was only 68, and most mistakes were neutral getting mixed up with
positive or negative.

**Four, noisy selection.** The laptop validation set has only 221 aspects, so the validation score jumps by up to
three points between checks. Which checkpoint we ended up keeping was partly luck.

### Slide 9 · Round 2 changes (8:45 – 10:00)
*Next slide.*

So for round two, each change answers one of these problems.

Against over-fitting, we added label smoothing, adversarial training on the word embeddings, and smaller learning
rates for the lower layers of the network.

For aspect localisation, we now mark the aspect directly in the sentence with hash symbols, like "the hash battery
life hash is disappointing". And the classifier now looks at the CLS token plus the average of the aspect's own
tokens, so it knows exactly which part of the sentence it is judging.

For noisy selection, we validate four times per epoch instead of twice.

For extraction, we added constrained decoding, so the model can never output an invalid tag sequence.

And finally, we tried a bigger model, RoBERTa-large.

One important detail: we also trained RoBERTa-base with the new recipe. Since it's the same model as in round one,
that tells us what the recipe does on its own, separate from what a bigger model does.

### Slide 10 · v1 → v2: sentiment (10:00 – 11:00)
*Next slide.*

And here are the results. Green means better than the best round-one model.

RoBERTa-large v2 is better in all four settings. On laptops it goes from 77.5 to 79.9 macro-F1, with accuracy going
from 80.4 to 83.2. On MAMS it goes from 84 to 85.1.

The biggest jump is in the hard direction, laptops to MAMS. The base model with the new recipe goes from 48.2 all the
way to 57.4. That's more than nine points.

We do want to be honest about one red number. The base v2 model is three points worse on MAMS to laptops. We'll come
back to why we think that happened.

### Slide 11 · v1 → v2: extraction and pipeline (11:00 – 11:45)
*Next slide.*

For extraction, the changes made little difference. Within a domain it was already strong, and laptop-to-MAMS
extraction actually got about six points worse.

But look at the bottom table, the full pipeline. Both v2 models improve the end-to-end score in all four settings.
So even where extraction didn't move, the better sentiment model carries the whole system forward.

Jeshwanth will take it from here.

---

## Part 3 · Jeshwanth (≈ 6.5 min, including the demo)

### Slide 12 · Learning dynamics (11:45 – 12:45)
*Next slide.*

Thanks, Varshit. Did round two actually fix the over-fitting? The answer is: partly.

On MAMS, yes. The gap between training and validation score at the chosen checkpoint dropped from about 15 points to
about 8 for the base model. You can also see in the bottom-right chart that the validation loss of the v2 models stays
low instead of climbing.

On laptops, it went the other way: the gap actually grew. Mostly that's because the round-one model happened to be
picked very early in training, before it had memorised much. With only two thousand training aspects and a tiny
validation set, the regularisation slows memorisation down but can't stop it. We think that's an honest limitation of
this dataset size.

### Slide 13 · Cross-dataset transfer (12:45 – 13:30)
*Next slide.*

This slide puts all six models side by side. Each little grid shows training data on the rows and test data on the
columns.

The pattern is the same everywhere, from TF-IDF all the way to RoBERTa-large. The bottom-left cell, trained on MAMS
and tested on laptops, is always much stronger than the top-right cell, trained on laptops and tested on MAMS.

So our main takeaway on generalisation is that the kind of data you train on matters more than the model. If you
train on sentences that mix opinions, the model learns to keep aspects apart, and that skill transfers.

### Slide 14 · Error analysis (13:30 – 14:45)
*Next slide.*

To understand what's still hard, we tagged every test sentence with simple rules: does it have a negation, a
contrast word like "but", several aspects, mixed polarities, no explicit opinion words, or an opinion word far away
from the aspect. Then we measured accuracy on each group.

The most interesting result is mixed polarities on laptops. These are sentences where one aspect is positive and
another is negative. In round one, that was the hardest group: RoBERTa got only 57% of them right. With the v2
recipe and the large model, it gets 74%. That's a gain of more than 16 points, while overall accuracy only went up by
about 3. For us, that's the clearest evidence that marking the aspect and pooling over it does exactly what we
designed it to do.

What's hard now is negation. For example, in "Did not enjoy the new Windows 8", the model still predicted positive.
Sarcasm and indirect statements are also still difficult.

### Slide 15 · Live demo (14:45 – 16:15)
*Switch to the notebook, section 9. Run the cell that loads the pipeline, then the examples cell.*

Let me show you the system working. This is our Jupyter notebook. The pipeline here uses two models chained together:
first the extractor finds the aspects, then the classifier assigns each one a sentiment.

*Run the examples cell.*

Here's the sentence from the very beginning of our presentation: "The laptop has an excellent display, but the battery
life is disappointing." And the system gives us: display, positive; battery life, negative. Exactly what we wanted.

Another one: "Customer support never answered my emails, although the screen is gorgeous." Customer support is
negative, the screen is positive. Notice it handles both the negation and the contrast here.

*Scroll to the last cell, type a new sentence and run it.* Let me type a new one live: "The keyboard feels cheap but
the processor handles video editing without any lag."

*Read the result out loud. If anything looks wrong, say so; it shows you understand the limits.*

So that's the full system, from raw text to aspect-level opinions.

*Switch back to the slides, slide 16.*

### Slide 16 · Conclusions (16:15 – 17:15)

To wrap up, here's what we learned.

First, fine-tuned Transformers beat TF-IDF by 14 to 17 macro-F1 points within a domain. Our best models reach 79.9
on laptops and 85.1 on MAMS.

Second, our second round of training, which was based directly on what we saw in the first-round curves, made
RoBERTa-large better in all four sentiment settings, and improved the full pipeline everywhere. The biggest gain was
on exactly the sentences we were targeting: mixed opinions.

Third, transfer depends on the shape of the training data more than on the model. And aspect extraction does not
survive a change of domain.

We also want to be upfront about the limitations. Each run used one random seed, so differences of one or two points
might not be significant. Our error tags are based on rules, not manual labels. And a few things got worse in round
two, like laptop-to-MAMS extraction.

If we continued, we'd run multiple seeds, pre-train further on unlabelled reviews from both domains, and try a
single model that does extraction and sentiment together.

### Slide 17 · Thank you (17:15 – 17:45)
*Next slide.*

Everything we showed today can be reproduced: the code, the notebook with all results, and the commands are on
this slide.

Thank you very much for watching. *(all three, if on camera)*

---

## Recording checklist
- [ ] Close notifications; set the screen to 1920×1080; slideshow mode for the deck.
- [ ] Run the notebook's section 9 once *before* recording so the demo models are cached (first run trains them for a
      few minutes).
- [ ] Do a 30-second test recording to check audio levels.
- [ ] If you record separately, keep each person's part as its own clip and join them in order; the hand-off lines at
      the end of Part 1 and Part 2 make the cuts feel natural.
- [ ] Aim for 17–18 minutes. If you run long, shorten the explanations on slides 4, 7 and 13 first.
