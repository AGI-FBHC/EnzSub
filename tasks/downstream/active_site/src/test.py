import torch, re
pid = "1A0I"
emb = torch.load(f"results/embeddings_multilayer/esm2_650m_base/test/{pid}.pt")
L_emb = emb[33].shape[0]

for line in open("../data/Enzyme_active_sites_test.txt"):
    if line.startswith(pid):
        label_str = line.strip().split(',', 2)[2]
        # 用正则只抓所有浮点数，彻底无视括号/空格/行尾杂字符
        nums = re.findall(r'-?\d+\.?\d*', label_str)
        labels = [float(x) for x in nums]
        break

pos = [i for i, v in enumerate(labels) if v > 0]
print("emb残基数:", L_emb)
print("标签长度:", len(labels))
print("长度是否相等:", L_emb == len(labels))
print("活性位点位置:", pos)