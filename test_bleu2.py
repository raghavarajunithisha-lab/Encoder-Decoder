import sacrebleu

all_preds = ["this is sentence one", "this is sentence two", "this is sentence three"]
all_refs = ["this is sentence one", "this is sentence two", "this is sentence three"]

try:
    res = sacrebleu.corpus_bleu(all_preds, [[r] for r in all_refs])
    print("Wrong format:", res.score)
except Exception as e:
    print("Wrong format error:", e)

try:
    res2 = sacrebleu.corpus_bleu(all_preds, [all_refs])
    print("Correct format:", res2.score)
except Exception as e:
    print("Correct format error:", e)
