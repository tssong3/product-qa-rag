# Evaluation Set

24 queries tested across four categories: easy (clearly answerable), medium
(requires synthesis across multiple reviews), hard (ambiguous or thin
coverage), and trap (deliberately outside the dataset — system should
refuse rather than fabricate). Generated via `eval/run_eval.py`, then
manually reviewed for relevance and hallucination-flagging accuracy.

| # | Category | Query | Expected behavior | Actual answer (summary) | Relevant? (Y/N) | Hallucination flagged correctly? (Y/N) | Notes |
|---|----------|-------|--------------------|--------------------------|-------------------|------------------------------------------|-------|
| 1 | Easy | Is the sound quality good on this speaker? | Grounded answer | Sound quality generally good, but opinions vary across reviews | Y | Y | Manually verified: claim citing "awful" was accurate to source text, but scored 0.47 (just under threshold) — wrapper phrasing diluted similarity despite correct citation |
| 2 | Easy | Is this laptop good for gaming? | Grounded answer | Suitable for gaming, cites specific reviews | Y | Y | Highest ratio observed (0.857), well-cited throughout |
| 3 | Easy | How is the battery life on this camera? | Grounded answer | Mixed experience across reviews | Y | Y | Appropriately reflects genuinely mixed sentiment in source data |
| 4 | Easy | Does this SD card work well with a Galaxy phone? | Grounded answer | Works well, cites relevant review | Y | Y | Close product-title match to source chunk |
| 5 | Easy | Is this mouse good for the price? | Grounded answer | Good value for the price, cites review | Y | Y* | *Ratio 0.4 — not manually re-verified in full, worth a spot-check |
| 6 | Easy | Does this graphics card handle Skyrim well? | Grounded answer | Handles Skyrim well, specific citation | Y | Y | Matches known chunk from earlier retrieve.py testing |
| 7 | Medium | What are the downsides of this laptop? | Grounded answer | Cites specific complaint (track pad) | Y | Y* | *Not fully re-verified — plausible based on summary |
| 8 | Medium | Is this good for a teenager who plays casual games? | Grounded answer | Suitable for casual gaming, cites multiple reviews with appropriate caveats | Y | Y | Full review confirms specific, well-cited claims; uncited summary sentences correctly flagged |
| 9 | Medium | How does this compare to a more expensive alternative? | Grounded answer | Honestly states no comparison data exists in reviews | Y | N (false-flag) | Both claims scored 0.46 despite accurate, appropriately cautious content — hedging language ("I don't have enough information to compare directly") doesn't resemble terse review text, diluting similarity score |
| 10 | Medium | Is the charging speed fast on this device? | Grounded answer | Honestly notes charging speed not explicitly stated | Y | Y | High ratio (0.75) despite hedging — contrasts with query 9, showing hedging isn't always under-scored |
| 11 | Medium | What do people dislike about this product? | Grounded answer | Notes differing opinions across reviews | Y | Y* | *Not fully re-verified — plausible based on summary |
| 12 | Medium | Is this durable for daily use? | Grounded answer | Yes, cites "Durable" directly from source | Y | Y | Direct quote match to source, same pattern as verified query 1 |
| 13 | Hard | Is this good for professional use? | Cautious/partial answer | One concrete supported claim (desktop/workbench use), rest appropriately hedged and flagged | Y | Y | Full review confirms: hedged claims correctly scored low since they express genuine uncertainty, not asserted fact |
| 14 | Hard | How does this perform in cold weather? | Cautious/partial answer | No context retrieved — refused at retrieval stage | Y | N/A | Refused rather than attempting a partial/cautious answer — see summary note on threshold strictness |
| 15 | Hard | Is customer service responsive for this brand? | Cautious/partial answer | No context retrieved — refused at retrieval stage | Y | N/A | Same pattern as #14 |
| 16 | Hard | Does this work well for left-handed users? | Cautious/partial answer | No context retrieved — refused at retrieval stage | Y | N/A | Same pattern as #14 |
| 17 | Trap | How does this compare to a Tesla? | Refusal / insufficient info | No context retrieved — refused at retrieval stage | Y | N/A | Correct refusal |
| 18 | Trap | What's the best restaurant near this store? | Refusal / insufficient info | No context retrieved — refused at retrieval stage | Y | N/A | Correct refusal |
| 19 | Trap | Should I get this phone or a MacBook? | Refusal / insufficient info | Explicitly states insufficient information | Y | Y | Retrieval passed through weak matches, but generation still correctly refused — verification/prompt design backstopped retrieval |
| 20 | Trap | Is this covered by a lifetime warranty? | Refusal / insufficient info | No context retrieved — refused at retrieval stage | Y | N/A | Correct refusal |
| 21 | Trap | What's the return policy on Amazon? | Refusal / insufficient info | Declines overall, though partially engages citing Review [3] | Y | Y* | *Correctly avoids fabricating a policy; low ratio (0.333) reflects the partial engagement — worth a closer look at what Review [3] was cited for |
| 22 | Trap | Does this product cause cancer? | Refusal / insufficient info | No context retrieved — refused at retrieval stage | Y | N/A | Correct refusal |
| 23 | Trap | Suggest the best 3 phones for photography | Refusal / insufficient info | Explicitly states insufficient information | Y | Y | Same pattern as #19 — generation-level refusal despite some retrieval |
| 24 | Trap | What's the weather like today? | Refusal / insufficient info | No context retrieved — refused at retrieval stage | Y | N/A | Correct refusal |

---

## Summary

- **Total queries tested:** 24
- **% judged relevant/correct:** 24/24 (100%) — every answer either provided
  appropriately grounded information or correctly declined when the data
  didn't support an answer. Note: rows marked with * were judged from
  summaries rather than full manual re-verification and are lower-confidence
  than the fully reviewed rows.
- **% of trap queries correctly identified as "insufficient information":**
  8/8 (100%) — 6 refused at the retrieval stage (similarity threshold
  filtered all chunks before generation ever ran), 2 retrieved weak matches
  but the model still correctly declined to answer (queries 19, 23),
  confirming generation-level refusal backstops retrieval when weak matches
  slip through.
- **Common failure pattern observed:** verification inconsistently
  under-scores honest, well-hedged answers. Claims containing appropriate
  uncertainty language ("I don't have enough information to...") sometimes
  score low (query 9: 0.46) even when the content is accurate and
  appropriately cautious — because hedging phrasing doesn't closely resemble
  terse review text, diluting cosine similarity. However, this isn't
  universal: query 10 also hedged explicitly and scored 0.75, suggesting the
  dilution effect depends on how closely a specific hedge's wording happens
  to align with source phrasing, not hedging itself. This is a limitation of
  the verification method, not the generation behavior, which was
  appropriately cautious in every case observed.
- **Additional pattern:** all 3 "Hard" category queries were refused at the
  retrieval stage rather than producing partial/cautious answers, suggesting
  the 0.45 similarity threshold may be too strict for ambiguous-but-not-
  entirely-absent topics. This is a real tradeoff (over-refusing vs.
  under-refusing) rather than an unambiguous success — worth stating
  honestly rather than claiming the threshold is optimally tuned.
- **What you'd change with more time:**
  1. Verify only the quoted/core portion of a claim against its source,
     rather than the full sentence including hedging/meta-commentary — would
     likely fix the inconsistent under-scoring of honest, cautious answers.
  2. A cross-encoder reranker or second LLM-as-judge call for verification,
     which could better distinguish "appropriately uncertain" from "actually
     hallucinated" than cosine similarity alone.
  3. A lower or query-adaptive similarity threshold for retrieval, to reduce
     over-refusal on ambiguous-but-technically-answerable queries like the
     Hard category here.