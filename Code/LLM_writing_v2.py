import datetime, json, re, httpx, os
from openai import OpenAI
from tqdm import tqdm


GOAL_MAP = {
    "Not sure": "0",
    "Introduce the topic": "1",
    "Describe the background": "2",
    "Establish the research problem": "3",
    "Specify the objectives": "4",
    "Summarize the findings": "5",
    "Map out the paper": "6",
}

SUPPORT_MAP = {
    "Weak": 1,
    "Moderate": 2,
    "Strong": 3,
}


def call_llm_label(title, paragraphs, model):
    prompt_template = f"""
    You are a professional academic reader tasked with labeling paragraphs in the Introduction section of a research paper. 
    Carefully read the title to understand the context of the paper. Then, evaluate the introduction paragraphs based on the authors' goals, which include:
    - Introduce the topic: Clearly communicates the subject's significance, often through an engaging hook.
    - Describe the background: Presents relevant background information or summarizes key research, including major issues and gaps.
    - Establish the research problem: Clarifies the focus of the research, highlighting the significance of the problem and how it differs from previous work.
    - Specify the objectives: Details what the authors aim to discover, including thesis statements or specific hypotheses.
    - Summarize the findings: Presents the outcomes.
    - Map out the paper: Provides an overview of the paper's structure, especially in non-standard formats.
    - Not sure: If a paragraph doesn’t align with these goals or covers multiple goals, classify it as "Not sure."
    Additionally, assign a degree of support for each label:
    - Strong: Clear and compelling evidence.  
    - Moderate: Some evidence, but not fully convincing.  
    - Weak: Little evidence to support the claim.
    For each paragraph, return your assessment in the following format without any other words: 
    [
        {{"paragraph": "..."[:5], "goal": "...", "support": "Strong | Moderate | Weak"}}, 
        ...
    ]
        """
    param_message = [{"role": "system", "content": prompt_template}]
    content = "Now, the title of the paper is: {0}; and paragraphs is: {1}.".format(title, paragraphs)
    content_message = [{"role": "user", "content": content}]
    param_message.extend(content_message)
    client = OpenAI(
        base_url="https://api.xty.app/v1",
        # 过期 sk-5waIHEUo9mqoF1Z3B3671eD882Ff40A093Fb94Fc5412A0E1
        api_key="sk-FPIygQsAqZ6jAdu8117b24F93f13421486E5Bf3f495c04B5",
        http_client=httpx.Client(follow_redirects=True))
    response = client.chat.completions.create(model=model, messages=param_message)
    return response.choices[0].message.content


def parse_llm_output(raw_text, n_paragraphs):
    print(f"n_paragraphs: {n_paragraphs}")
    try:
        data = json.loads(raw_text)
        # if len(data) != int(n_paragraphs):
        #     return None
        labels = {}
        for i, item in enumerate(data, start=1):
            # llm返回数字类型
            # goal = int(item["goal"])
            # weight = int(item["support"])
            # if goal not in range(0, 7):
            #     goal = 0
            # if weight not in (1, 2, 3):
            #     weight = 1
            # labels[i] = {
            #     "goal": str(goal),
            #     "weight": weight
            # }
            # llm返回字符串类型
            goal_text = item.get("goal", "Not sure")
            support_text = item.get("support", "Weak")
            goal = GOAL_MAP.get(goal_text, "0")
            weight = SUPPORT_MAP.get(support_text, 1)
            labels[i] = {
                "goal": goal,
                "weight": weight
            }
        return labels
    except Exception as e:
        print("[PARSE ERROR]", e)
        return None


def input_text(field, file_path, model):
    # 1. 读取 title
    paper_title = {}
    with open(file_path + f"{field}_journal_paper_reference_6paras.txt", "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if parts[0] == "s2orc_paper_id":
                continue
            paper_title[parts[0]] = parts[-1]
    print(f"[INFO] {field}: title loaded.")
    # 2. 输出文件路径
    ws_path = file_path + f"{field}_writing_score_{model}.txt"
    # 3. 断点续传
    processed_ids = set()
    if os.path.exists(ws_path):
        with open(ws_path, "r") as f:
            next(f, None)  # skip header
            for line in f:
                pid = line.split(";")[0]
                processed_ids.add(pid)
    print(f"[INFO] {field}: {len(processed_ids)} papers already processed.")
    # >>> 4. 预读 introduction，计算总任务数
    intro_items = []
    with open(file_path + f"{field}_intro_text.txt", "r") as fintro:
        for line in fintro:
            parts = line.strip().split("-----", 1)
            if len(parts) != 2:
                continue
            pid, intro = parts
            if pid in paper_title:
                intro_items.append((pid, intro))
    total_to_process = sum(
        1 for pid, _ in intro_items if pid not in processed_ids
    )
    print(f"[INFO] {field}: {total_to_process} papers to process.")
    # 4. 打开输出文件-append模式
    write_header = not os.path.exists(ws_path)
    with open(ws_path, "a") as fws:
        if write_header:
            fws.write("s2orc_paper_id;writing_score\n")
        # 5. 读取 introduction
        with tqdm(total=total_to_process, desc=f"{field} | LLM labeling") as pbar:
            for pid, intro_text in intro_items:
                if pid in processed_ids:
                    continue
                paragraphs = [p.strip() for p in intro_text.split(";;;;;") if p.strip()]
                if len(paragraphs) < 2:
                    pbar.update(1)
                    continue
                title = paper_title[pid]
                # 6. 调用 LLM 打标
                try:
                    raw_output = call_llm_label(title, paragraphs, model)
                except Exception as e:
                    print(f"[ERROR] API failed for {pid}: {e}")
                    pbar.update(1)
                    continue
                label_dict = parse_llm_output(raw_output, len(paragraphs))
                if not label_dict:
                    print(f"[WARN] LLM parse failed: {pid}")
                    pbar.update(1)
                    continue
                # 7. 写 writing_score
                ws_items = [
                    f"{label_dict[i]['goal']}:{label_dict[i]['weight']}"
                    for i in range(1, len(label_dict) + 1)
                ]
                fws.write(f"{pid};{','.join(ws_items)}\n")
                processed_ids.add(pid)
                pbar.update(1)
    print(f"[DONE] {field}: all papers processed.")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # gpt-5.2 / gemini-3-flash-preview-nothinking
    for field in ["economics", "math", "medicine", "physics", "cs"]:
        input_text(f"{field}", '/Volumes/SanDisk_2TB/s2orc_field/test_set/', 'gpt-5.2')
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done LLMs_v2!')
