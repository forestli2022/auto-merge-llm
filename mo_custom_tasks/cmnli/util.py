
def doc_to_text(doc):
    return "语句一：\"" + doc['sentence1'] + "\"\n语句二：\"" + doc['sentence2'] + "\"\n请问这两句话是什么关系？"


 def process_label(doc, doc):
    label = doc["label"]
    label_map = {"contradiction": "矛盾", "neutral": "无关", "entailment": "蕴含"}
    return label_map[lable]