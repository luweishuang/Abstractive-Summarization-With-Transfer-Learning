import jsonlines
import os


index = 1
cc_num = 0
content_file_path = "data/eval_story.txt"
summary_file_path = "data/eval_summ.txt"

with open("../data/Newsroom/test.label.info.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        summary_list = item.get("summary", [])
        text_list = item.get("text", [])
        if len(summary_list) > 0 and len(text_list) > 0:
            summary_str = "".join(summary_list)
            content_str = "".join(text_list)
            # print(summary_str)
            # print("----------------------------------")
            # print(content_str)
            # print("**********************************")
            pos = summary_str.find("<?xml:")
            if pos >= 0:
                continue
            if content_str.find(summary_str) >= 0:
                cc_num += 1

            index += 1
            with open(content_file_path, "a+") as fw:
                fw.write(content_str.replace("\n", "."))
            with open(summary_file_path, "a+") as fw:
                fw.write(summary_str.replace("\n", "."))
        else:
            print(" summary or content is null !")
print("summary in content num = %d " % cc_num)          # 93352
print("total txt num = %d " % index)

# train
# summary in content num = 93352
# total txt num = 993403
# test
# summary in content num = 10156
# total txt num = 108704
