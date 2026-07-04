# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Framework to use in the project

- OODA Framework
- L99 Level
- UltraThink
- Godmode
- SOLID Model
- Python

## 6. Project Porpouse
build a RAG for military and defence to use with a local llm (lm studio) in python step by step using links below for citations.

## 7. Link for RAG Documents
The Dark Future of Next-Gen Asymmetric Warfighting: https://arxiv.org/pdf/2408.12045

ANALYSIS AND ASSESSMENT OF GATEWAY PROCESS: https://www.cia.gov/readingroom/docs/CIA-RDP96-00788R001700210016-5.pdf

MCDP 1 Warfighting: https://upload.wikimedia.org/wikipedia/commons/4/4e/MCDP_1_Warfighting.pdf?utm_source=en.wikisource.org&utm_campaign=index&utm_content=original

On War: https://www.gutenberg.org/cache/epub/1946/pg1946.txt

The Art of War: https://www.gutenberg.org/cache/epub/132/pg132.txt

FM 3-0: https://soldat-und-technik.de/wp-content/uploads/2022/10/ARN36290-FM_3-0.pdf

FM 5-0: https://stephengates.com/ADM/FM-JUL22.pdf

ATP_2-01.3_Intelligence_Preparation_of_the_Battlefield: https://home.army.mil/wood/application/files/8915/5751/8365/ATP_2-01.3_Intelligence_Preparation_of_the_Battlefield.pdf

FM 2-0: https://www.bits.de/NRANEU/others/amd-us-archive/fm2-0fd%2809%29.pdf

Army Techniques Publication (ATP) 3-60: https://irp.fas.org/doddir/army/atp3-60.pdf

FM 3-06 URBAN OPERATIONS: https://irp.fas.org/doddir/army/fm3-06.pdf

FM 3-12 Cyberspace and Electromagnetic Warfare: https://irp.fas.org/doddir/army/fm3-12.pdf

ADP3: https://www.bits.de/NRANEU/others/amd-us-archive/ADP3-0%2816%29.pdf

Operations Research, Second Edition: https://www.bbau.ac.in/dept/UIET/EME-601%20Operation%20Research.pdf

Data Structures and Algorithms: https://mta.ca/~rrosebru/oldcourse/263114/Dsa.pdf

Corey Trevena - Pathfinding Algorithms in Navigational Meshes PDF: https://www.cs.csustan.edu/~mmartin/teaching/CS4960S15/Corey%20Trevena%20-%20Pathfinding%20Algorithms%20in%20Navigational%20Meshes%20PDF.pdf

Dispensa Game Theory: https://didattica.unibocconi.it/mypage/upload/48808_20220802_072515_02.08.2022TEXTBOOKGTAST_COMPRESSED.PDF

fm-3-09-fire-support-and-field-artillery-operations: https://www.revista-artilharia.pt/admin/upload/ficheiros/ficheirosMultimedia/fm-3-09-fire-support-and-field-artillery-operations.pdf

FM 3-24 INSURGENCIES AND COUNTERING INSURGENCIES: https://irp.fas.org/doddir/army/fm3-24.pdf

AJP-3.3 ALLIED JOINT DOCTRINE FOR AIR AND SPACE OPERATIONS: https://www.coemed.org/files/stanags/01_AJP/AJP-3.3_EDB_V1_E_3700.pdf

AJP-3.20, Allied Joint Doctrine for Cyberspace Operations: https://assets.publishing.service.gov.uk/media/5f086ec4d3bf7f2bef137675/doctrine_nato_cyberspace_operations_ajp_3_20_1_.pdf

Lanchester-Type Models of Warfare: https://apps.dtic.mil/sti/tr/pdf/ADA090842.pdf

Military Operations Research. Summer 1994: https://apps.dtic.mil/sti/tr/pdf/ADA321335.pdf

Mathematical Models for Planning in Military and Humanitarian Logistics: https://publications.tno.nl/publication/34644360/brDxfOYa/wagenvoort-2025-mathematical.pdf

A GIS-BASED SIMULATION MODEL FOR MILITARY PATH PLANNING OF UNMANNED GROUND ROBOTS: https://www.witpress.com/Secure/ejournals/papers/SSE010302f.pdf

Military route planning in battlefield simulation: effectiveness problems and potential solutions: http://www.tarapata.strefa.pl/publikacje/jtit_2003.pdf


