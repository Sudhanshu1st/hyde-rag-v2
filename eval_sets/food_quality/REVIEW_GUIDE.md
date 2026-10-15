# Reviewing the food-quality questions

`review.csv` holds machine-drafted candidate questions. Only rows you accept or edit go into
the eval set, so every question in it has been checked by a person. It opens in Excel.

## For each drafted row, fill `decision`

| decision | When | What else to fill |
|---|---|---|
| `accept` | The question is clear and realistic, the reference answer is right, and the evidence quote supports it | nothing |
| `edit` | It's fixable | `edited_question` and/or `edited_answer`. To fix the evidence, replace `evidence_quote` with a verbatim quote from the document |
| `reject` | Vague, trivial, wrong, a duplicate, or not something a QA manager would ask | optional `notes` |

Checks per row:
- **Answerable rows:** does the quote actually contain the answer? `evidence_located = False`
  means the drafted quote wasn't found in the document text. Fix the quote, or reject the row.
- **Unanswerable rows:** is it really unanswerable from the corpus? If you know the corpus
  answers it, reject it.
- **Numeric rows:** check the unit (°F vs °C, hours vs minutes).
- **Jargon rows** are deliberately short keyword queries (e.g. "CCP for cooling?").

## Write about 30 of your own (`hum-...` rows at the bottom)

Fill in:
- `question`;
- `answerable` (`yes` or `no`);
- `reference_answer`;
- `evidence_doc` (a `doc_id` from `data_sources.yaml`);
- `evidence_quote`: words copied exactly from that document. Separate several quotes with `||`.

Questions you write are reported as their own subset.

## Afterwards

```
uv run hyde-rag build-evalset
```

This writes `dev.jsonl` (30%) and `test.jsonl` (70%), split by question type. It also lists any
answerable row whose quote couldn't be found, so you can fix it and re-run.
