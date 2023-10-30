from evaluate import load as load_evaluator

# bertscore_evaluator = load_evaluator("bertscore")
bertscore_evaluator = load_evaluator("bleu")


predicted_captions = ["Hello there, how are you", "Table under the tree"]

val_captions = [
    ["Hello there, how are you"], 
    ["Table under the tree right"]
    ]
# get bertscore
# bertscores = bertscore_evaluator.compute(predictions=predicted_captions, references=val_captions, model_type="distilbert-base-uncased", lang="en", verbose=True)
bertscores = bertscore_evaluator.compute(predictions=predicted_captions, references=val_captions)


# print('precision ', bertscores['precision'])
# print('recall ', bertscores['recall'])
# print('f1 ', bertscores['f1'])


print('bleu ', bertscores['bleu'])
print('precisions ', bertscores['precisions'])

