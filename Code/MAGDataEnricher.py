import pandas as pd
import os


class MAGDataEnricher:
    def __init__(self):
        self.mag_root = "/path/to/MAG/"
        self.s2orc_root = "/path/to/s2orc_fields/"
        self.output_root = "/path/to/output/"
        os.makedirs(self.output_root, exist_ok=True)
        self._load_mag_tables()
        self._prepare_field_size()

    # 1. 加载 MAG 各主表
    def _load_mag_tables(self):
        print("📂 Loading MAG tables...")
        self.papers = pd.read_csv(
            os.path.join(self.mag_root, "Papers.tsv"), sep='\t',
            usecols=["PaperId", "Year", "CitationCount", "ReferenceCount"]
        )
        self.paper_authors = pd.read_csv(
            os.path.join(self.mag_root, "PaperAuthorAffiliations.tsv"), sep='\t',
            usecols=["PaperId", "AuthorId", "AuthorSequenceNumber"]
        )
        self.authors = pd.read_csv(
            os.path.join(self.mag_root, "Authors.tsv"), sep='\t',
            usecols=["AuthorId", "PaperCount", "CitationCount"]
        )
        self.paper_fields = pd.read_csv(
            os.path.join(self.mag_root, "PaperFieldsOfStudy.tsv"), sep='\t',
            usecols=["PaperId", "FieldOfStudyId"]
        )
        print("✅ MAG data loaded successfully.")

    # 2. 计算每个领域的论文数量（topic_size）
    def _prepare_field_size(self):
        print("🧮 Computing field sizes (topic_size)...")
        field_size = (
            self.paper_fields["FieldOfStudyId"]
            .value_counts()
            .rename_axis("FieldOfStudyId")
            .reset_index(name="topic_size")
        )
        self.field_size = field_size
        print(f"✅ Computed {len(field_size)} unique fields.")

    # 3. h-index 计算函数
    @staticmethod
    def compute_h_index(citations):
        citations_sorted = sorted(citations, reverse=True)
        h = 0
        for i, c in enumerate(citations_sorted):
            if c >= i + 1:
                h = i + 1
            else:
                break
        return h

    # 4. 处理单个领域
    def process_domain(self, domain: str):
        input_path = os.path.join(self.s2orc_root, f"{domain}_journal_paper_reference_6paras.txt")
        if not os.path.exists(input_path):
            print(f"⚠️ {input_path} not found. Skipping {domain}.")
            return
        print(f"\n📘 Processing domain: {domain}")
        df = pd.read_csv(input_path, sep=';', dtype={'mag_id': str})
        df.rename(columns={'mag_id': 'PaperId'}, inplace=True)
        df = df.merge(self.papers, on=["PaperId", "Year"], how="left", suffixes=('', '_MAG'))
        merged_authors = self.paper_authors.merge(self.authors, on="AuthorId", how="left")
        first_authors = merged_authors.loc[merged_authors["AuthorSequenceNumber"] == 1]
        first_agg = first_authors.groupby("PaperId").agg(
            first_author_prod=("PaperCount", "max"),
            first_author_citations=("CitationCount", "max")
        ).reset_index()
        last_authors = merged_authors.loc[
            merged_authors.groupby("PaperId")["AuthorSequenceNumber"].transform("max")
            == merged_authors["AuthorSequenceNumber"]
        ]
        last_agg = last_authors.groupby("PaperId").agg(
            last_author_prod=("PaperCount", "max"),
            last_author_citations=("CitationCount", "max")
        ).reset_index()
        df = df.merge(first_agg, on="PaperId", how="left")
        df = df.merge(last_agg, on="PaperId", how="left")
        df["first_author_h_index"] = (df["first_author_citations"] / 100).fillna(0).astype(int)
        df["last_author_h_index"] = (df["last_author_citations"] / 100).fillna(0).astype(int)
        df["title_len"] = df["title"].astype(str).apply(lambda x: len(x.split()))
        # ---- 计算 topic_size ----
        topic_info = self.paper_fields.merge(self.field_size, on="FieldOfStudyId", how="left")
        topic_max = topic_info.groupby("PaperId")["topic_size"].max().reset_index()
        df = df.merge(topic_max, on="PaperId", how="left")
        output_file = os.path.join(self.output_root, f"{domain}_paper_author_topic.txt")
        df.to_csv(output_file, index=False)
        print(f"✅ Saved enriched data: {output_file} (n={len(df)})")

    # 5. 批处理所有领域
    def process_all_domains(self, domains):
        for d in domains:
            self.process_domain(d)
        print("\n🎉 All domains processed successfully!")

    # 6. 新增功能：提取期刊信息并计算 journal_cc
    def extract_journal_info(self, domain: str):
        input_path = os.path.join(self.s2orc_root, f"{domain}_journal_paper_reference_6paras.txt")
        if not os.path.exists(input_path):
            print(f"⚠️ {input_path} not found. Skipping {domain}.")
            return
        print(f"\n📗 Extracting journal info for domain: {domain}")
        df = pd.read_csv(input_path, sep=';', dtype={'mag_id': str})
        df.rename(columns={'mag_id': 'PaperId'}, inplace=True)
        # 取论文所属期刊
        papers_journal = self.papers[["PaperId", "JournalId", "CitationCount"]].dropna(subset=["JournalId"])
        # 每个 journal 的总引用数
        journal_cc = (
            papers_journal.groupby("JournalId")["CitationCount"]
                .sum()
                .reset_index()
                .rename(columns={"CitationCount": "journal_cc"})
        )
        # 合并 journal_id
        df = df.merge(papers_journal[["PaperId", "JournalId"]], on="PaperId", how="left")
        df = df.merge(journal_cc, on="JournalId", how="left")
        # 格式化输出
        df_out = df.rename(columns={
            "PaperId": "mag_id",
            "s2orc_paper_id": "s2orc_paper_id" if "s2orc_paper_id" in df.columns else "s2orc_paper_id"
        })[["s2orc_paper_id", "mag_id", "JournalId", "journal_cc"]]
        df_out.rename(columns={"JournalId": "journal_id"}, inplace=True)
        output_file = os.path.join(self.output_root, f"{domain}_journal_cc.txt")
        df_out.to_csv(output_file, sep=';', index=False)
        print(f"✅ Saved journal info: {output_file} (n={len(df_out)})")


if __name__ == "__main__":
    enricher = MAGDataEnricher()
    domains = ["Biology", "Medicine", "Physics", "Mathematics", "ComputerScience", "Economics"]
    enricher.process_all_domains(domains)
    for field in domains:
        enricher.extract_journal_info(field)
