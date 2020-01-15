# import jsonlines
#
# index = 1
# cc_num = 0
# content_file_path = "data/eval_story.txt"
# summary_file_path = "data/eval_summ.txt"
#
# with open("../data/Newsroom/test.label.info.jsonl", "r+", encoding="utf8") as f:
#     for item in jsonlines.Reader(f):
#         summary_list = item.get("summary", [])
#         text_list = item.get("text", [])
#         summary_str = "".join(summary_list)
#         content_str = "".join(text_list)
#         # if 'low' == item.get("compression_bin", ""):
#         if len(content_str.split(" ")) < 256:
#             cc_num += 1
#             print(summary_str)
#             print("----------------------------------")
#             print(content_str)
#             print("**********************************")
#             index += 1
#         else:
#             print(" summary or content is null !")
# print("summary in content num = %d " % cc_num)          # 93352
# print("total txt num = %d " % index)

# train
# summary in content num = 93352
# total txt num = 993403
# test
# summary in content num = 10156
# total txt num = 108704



from newsroom import jsonl

# Read file entry by entry:
with jsonl.open("../data/Newsroom/dev.jsonl.gz", gzip=True) as train_file:
    for entry in train_file:
        print("************************************")
        print(entry["summary"])
        print("-------------------------------")
        print(entry["text"])