# 설치 필요: pip install python-frontmatter markdownify tqdm
import re
from pathlib import Path
import frontmatter
from markdownify import markdownify as md2text
from tqdm import tqdm

def sub_code_blocks(text: str) -> str:
    # fenced code blocks (non-greedy, DOTALL)
    text = re.sub(r"``````", "", text)
    # inline code
    text = re.sub(r"`[^`]*`", "", text)
    return text

def sub_links_and_wikilinks(text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # 이미지
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)   # 링크
    text = re.sub(r"\[\[(?:.*?\|)?([^\]]+)\]\]", r"\1", text)  # wikilink
    text = re.sub(r"<[^>]+>", "", text)  # html tags
    return text

def sub_markdown_markers(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"\|", "", text)  # table pipes
    text = re.sub(r"[-*_]{3,}", "", text)  # hr lines
    text = re.sub(r"\*\*", "", text)  # bold markers
    text = re.sub(r"{{", "", text)
    text = re.sub(r"}}", "", text)
    text = re.sub(r"\*", "", text)
    return text

def sub_latex_and_math(text: str) -> str:
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)  # $$...$$
    text = re.sub(r"\$.*?\$", "", text)  # $...$
    text = re.sub(r"\\\([^\)]*?\\\)", "", text)  # \(...\)
    text = re.sub(r"\\\[.*?\\\]", "", text)  # \[...\]
    text = re.sub(r"\\begin\{[^\}]+\}[\s\S]*?\\end\{[^\}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*\}|\\[a-zA-Z]+", "", text)
    return text

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def to_plaintext_via_markdownify(md: str) -> str:
    # markdownify: converts markdown -> readable plaintext (tables/links handled)
    try:
        return md2text(md)
    except Exception:
        return md  # fallback

def load_obsidian_notes(vault_path):
    p = Path(vault_path)
    md_files = list(p.rglob("*.md"))
    docs = []
    for f in tqdm(md_files, desc="Reading md files"):
        try:
            post = frontmatter.load(f)
            content = post.content
            # 1) convert markdown to plaintext as first pass (handles some tables nicely)
            content = to_plaintext_via_markdownify(content)
            # 2) remove code/math/latex and other markup
            try:
                content = sub_code_blocks(content)
                content = sub_latex_and_math(content)
                content = sub_links_and_wikilinks(content)
                content = sub_markdown_markers(content)
            except re.error as e:
                 print(f"[REGEX ERROR] during subping file: {f} -> {e}")
                 continue # Skip this file if regex fails during subping
            content = normalize_whitespace(content)
            # optional: if file is extremely short after cleaning, skip or note it
            if not content:
                continue
            docs.append({"path": str(f.relative_to(p)), "text": content, "meta": dict(post.metadata)})
        except re.error as e:
            print(f"[REGEX ERROR] during frontmatter load file: {f} -> {e}")
        except Exception as e:
            print(f"[OTHER ERROR] file: {f} -> {e}")
    return docs

# 사용 예
#vault = "C://Users//User//OneDrive - konkuk.ac.kr//문서//n8n_metacog"
#docs = load_obsidian_notes(vault)
#print(f"Loaded {len(docs)} documents.")
#print(docs[1]["path"], docs[1]["text"][:100])  # 첫 문서의 경로와 앞 100자 보기