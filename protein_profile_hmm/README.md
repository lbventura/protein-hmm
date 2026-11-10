# Protein profile HMM

## Modules and execution

- `protein_profile_hmm.py`: `ProfileHMM.build_from_msa` constructs a profile using counts and pseudocount smoothing; `viterbi` scores the best alignment path, while `forward` sums over allowed paths.
- `amino_acids.py`: canonical amino-acid alphabet, residue encoding, and background frequencies used in scoring.
- `msa.py`: Stockholm alignment parsing and selection of consensus columns using `symfrac`, the minimum fraction of canonical residues in a column. This measures residue occupancy, not agreement on a single amino acid.
- `fasta_parser.py`: reads FASTA records as names and protein sequences.
- `protein_demo.py`: builds a globin profile and compares scores for globins and an unrelated tyrosine kinase.
- `download_data.py`: fetches HMMER tutorial datasets and converts the Swiss-Prot negative control to FASTA.

After following the [development setup](../README.md#set-up-a-development-environment), run from the repository root:

```bash
python -m protein_profile_hmm.protein_demo
python -m pytest tests/test_profile_hmm.py tests/test_globins_integration.py
```

To download fresh copies of the [datasets](../data/DATASETS.md), overwriting the existing files:

```bash
python -m protein_profile_hmm.download_data
```

## Profile architecture

Each of the `K` consensus positions has a Match state (`M_k`) and a silent Delete state (`D_k`). Insert states (`I_k`) represent extra residues between consensus positions, including boundary slots. The Begin state (`B`) is silent.

- **Match:** consumes a residue aligned to consensus position `k`; the residue need not equal the most common amino acid there.
- **Insert:** consumes an extra residue without advancing the consensus position.
- **Delete:** advances past a consensus position without consuming a residue.

The transition array stores nine types per slot:

| Index | Transition | Meaning |
| --- | --- | --- |
| 0 | M_k → M_{k+1} | Advance to the next match |
| 1 | M_k → I_k | Start an insertion |
| 2 | M_k → D_{k+1} | Start a deletion |
| 3 | I_k → M_{k+1} | End an insertion |
| 4 | I_k → I_k | Extend an insertion |
| 5 | D_k → M_{k+1} | End a deletion |
| 6 | D_k → D_{k+1} | Extend a deletion |
| 7 | B → M_1 | Begin at the first match (slot 0 only) |
| 8 | B → D_1 | Begin by deleting the first position (slot 0 only) |

Scoring spans the full profile, allowing deletions, and consumes the query residues; boundary Insert states accommodate flanking residues. Emission and transition probabilities are stored in log2 space. Scores are log-odds in bits relative to a background model: positive scores favor the profile over that background, but are not calibrated significance estimates or HMMER E-values.

## Following profile construction

`build_from_msa` collects observations, adds priors, normalizes the distributions, and stores log probabilities. The counting stage has four responsibilities:

1. `_count_alignment` determines match columns from occupancy across all sequences. It also manages the 250,000-cell batch budget; every batch uses the same column assignments.
2. `_trace_alignment` represents each sequence as a raw path through Begin, Match, Insert, and Delete states. A residue in a match column visits M; a gap visits D. A residue in an insert column visits I; a gap in that column contributes no event.
3. `_repair_trace` replaces adjacent I/D or D/I events with one Match, retaining the Delete position and Insert residue. Pairs are selected from left to right within each sequence; each event participates in at most one pair.
4. `_count_profile` counts emissions and transitions from the shared repaired trace. Match/Insert events contribute residues; consecutive events from the same sequence contribute transitions. `np.add.at` adds observations directly into the two-dimensional count arrays, including repeated visits to a slot.

For example, with match columns 1 and 4:

```text
Alignment       Training path
AGGC            B -> M1(A) -> I1(G) -> I1(G) -> M2(C)
A..C            B -> M1(A) -----------------> M2(C)
A...            B -> M1(A) -----------------> D2
```

The G residues both add one to insertion slot 1's G count. The second row contributes M1 -> M2 across the skipped insert gaps. The third row contributes M1 -> D2, with no residue emitted by D2.

An alignment can also imply an edge absent from the scoring graph. For
`["AG.E", "A.CE", "A.CE"]`, the first row needs repair:

```text
Column assignments:  M1 I1 M2 M3
Raw path:       B -> M1(A) -> I1(G) -> D2 -> M3(E)
Repaired path:  B -> M1(A) -> M2(G) -> M3(E)
```

G now contributes to Match position 2, and both surrounding edges are counted
as Match-to-Match transitions. Repair keeps the emitted sequence and the
globally selected match columns, including K, unchanged. For overlapping
events, `D2 -> I2(G) -> D3` becomes `M2(G) -> D3`; the Insert is consumed by
the first pair. `_count_transitions` rejects unsupported edges if passed an
unrepaired trace directly. Construction repairs once before both counters run.

`AlignmentTrace` stores parallel arrays with one entry per event:

| Field | Meaning |
|---|---|
| `sequence` | Zero-based row index within the batch; prevents transitions across rows |
| `state` | A `State` code: BEGIN, MATCH, INSERT, or DELETE |
| `position` | Match/Delete position 1..K, Insert slot 0..K, or 0 for Begin |
| `residue` | Encoded amino acid for Match/Insert; -1 for silent events |

## Comparing the construction stages with HMMER

HMMER also selects match columns, derives training traces, and counts those traces into an HMM. Its `matassign2hmm` routine calls `p7_trace_FauxFromMSA`, then repairs and validates the traces before calling `p7_trace_Count`. See the pinned upstream [model builder](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/src/build.c) and [trace implementation](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/src/p7_trace.c).

Here, `_trace_alignment` provides the path-construction stage, `_repair_trace`
follows HMMER's left-to-right pair-repair convention, and
`_count_emissions`/`_count_transitions` count the repaired paths.

## Following query scoring

Both scorers consume one query residue at a time. `_init_score_rows` encodes the
query, starts Begin at zero, and fills the leading deletion path. Each row has
K+1 entries: Match/Delete positions 1..K and Insert slots 0..K.

For each residue, `_viterbi_row` or `_forward_row` performs three updates:

1. Match scores read the previous row's M/I/D predecessors. M1 uses Begin or
   I0 explicitly; positions M2..MK are computed together with array operations.
2. Insert scores read the previous row's M/I scores at the same slot. These
   updates also run across all positions together.
3. Delete scores advance from left to right in the current row. D_k needs the
   newly computed D_(k-1), so this sweep remains ordered.

Viterbi selects the highest candidate score. Forward sums path weights using
`np.logaddexp2`, preserving base-2 units and the unreachable-state checks.
Both return a scalar score after combining M_K, I_K, and D_K.

For example:

```python
from protein_profile_hmm.msa import MSA
from protein_profile_hmm.protein_profile_hmm import ProfileHMM

msa = MSA(names=["s1", "s2", "s3"], rows=["AGGCE", "A..CE", "A...E"])
profile = ProfileHMM.build_from_msa(msa)
print(f"{profile.viterbi('AAAA'):.3f}")  # -2.181 bits
```

See [first-residue walkthrough](viterbi_first_residue_walkthrough.md) and its
follow-ups for a thorough step-by-step explanation of this example.
