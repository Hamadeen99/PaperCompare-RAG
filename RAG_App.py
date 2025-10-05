import os, re
from typing import List, Dict
import PyPDF2
import chromadb  
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


chroma_DB = "./paper_index"
emb_model = "BAAI/bge-small-en-v1.5"
llm_model = "microsoft/Phi-3-mini-4k-instruct"

top_k= 4          # context per paper
axis_k = 3         # chunks per subquery 
chars = 1200        # 1200 char per-paper 
bullets = 4        # bullets points in the answer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


embedding = SentenceTransformer(emb_model, device=device)

tok = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    llm_model,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="eager",   
    device_map="auto",
)
tok.pad_token = tok.eos_token
llm.eval()
torch.set_grad_enabled(False)

try:
    client = chromadb.PersistentClient(path=chroma_DB)
except Exception:
    client = chromadb.Client()
col = client.get_or_create_collection(name="ai_papers")


p_chunk: Dict[str, List[Dict]] = {}   

# PDF to Text to Chunks
paper_sec = [
    "abstract", "introduction", "method", "methods", "approach",
    "experiment", "experiments", "result", "results",
    "conclusion", "limitations", "limitation", "dataset", "data"
]

def read_pdf_text(pdf_path: str) -> str:
    pages = []
    with open(pdf_path, "rb") as f:
        r = PyPDF2.PdfReader(f)
        for p in r.pages:
            t = p.extract_text() or ""
            pages.append(t)
    raw = " ".join(pages)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw

def smart_chunks(text: str, paper_id: str) -> List[Dict]:
    sents = text.split(". ")
    chunks, buf, sec = [], [], "general"
    for i, s in enumerate(sents):
        sl = s.lower()
        
        hit = [name for name in paper_sec if name in sl and len(s.split()) < 12]
        if hit:
            sec = hit[0]
        buf.append(s.strip() + ".")
        
        if sum(len(x) for x in buf) > 650 or (i % 4 == 0 and i > 0):
            block = " ".join(buf).strip()
            chunks.append({
                "text": block, "paper_id": paper_id, "section": sec,
                "chunk_id": f"{paper_id}_{len(chunks)}"
            })
            buf = []
    if buf:
        chunks.append({
            "text": " ".join(buf).strip(), "paper_id": paper_id, "section": sec,
            "chunk_id": f"{paper_id}_{len(chunks)}"
        })
    return chunks


def add_chunks_to_chroma(chunks: List[Dict]):
    docs = [c["text"] for c in chunks]
    embs = embedding.encode(docs, normalize_embeddings=True, convert_to_numpy=True).tolist()
    col.add(
        embeddings=embs,
        documents=docs,
        metadatas=chunks,
        ids=[c["chunk_id"] for c in chunks],
    )

def chroma_topk(query: str, paper_id: str, k: int = top_k) -> List[Dict]:
    q = embedding.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()
    res = col.query(
        query_embeddings=q,
        n_results=max(k, 8),
        where={"paper_id": paper_id},
        include=["documents", "metadatas", "distances"],  # keys in Chroma
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for i in range(min(k, len(docs))):
        out.append({"text": docs[i], "metadata": metas[i]})
    return out


trainKW = ["batch size", "learning rate", "epochs", "steps", "iterations", "schedule", "warmup", "dropout"]

def training_subqueries(question: str) -> List[str]:
    return [f"{kw}: {question}" for kw in trainKW]

def axis_topk(question: str, paper_id: str, per_axis: int = axis_k , final_k: int = top_k) -> List[Dict]:
    ql = question.lower()
    subqs = []
    if any(x in ql for x in ["batch", "epoch", "step", "learning rate", "lr", "warmup", "dropout"]):
        subqs.extend(training_subqueries(question))
    
    subqs.append(question)

    hits, seen = [], set()
    for subq in subqs[:5]:
        for h in chroma_topk(subq, paper_id, k=per_axis):
            sig = h["text"][:120]
            if sig not in seen:
                seen.add(sig)
                hits.append(h)
            if len(hits) >= final_k:
                break
        if len(hits) >= final_k:
            break
    return hits[:final_k]


# Prompt 
def make_contrast_prompt(question: str, chunks_a: List[Dict], chunks_b: List[Dict],
                         max_ctx_chars: int = chars, bullets: int = bullets) -> str:
    ctx_a = "\n".join(c["text"] for c in chunks_a)[:max_ctx_chars]
    ctx_b = "\n".join(c["text"] for c in chunks_b)[:max_ctx_chars]

    return f"""<|system|>
You compare two AI papers. Answer only from the provided contexts for Paper A and Paper B.
Output exactly 4 bullets in this order and format:
- **batch size** — A: <fact>; B: <fact>
- **steps/epochs** — A: <fact>; B: <fact>
- **learning rate** — A: <fact>; B: <fact>
- **warmup/other training detail** — A: <fact>; B: <fact>
Rules:
- If a value is present numerically in context, do not answer "not stated".
- If truly absent, answer exactly "not stated".
- Be <16 words per side, factual, specific, no filler. No citations.
<|end|>
<|user|>
Question: {question}

Paper 1 context:
{ctx_a}

Paper 2 context:
{ctx_b}
<|end|>
<|assistant|>"""


def generate_answer(prompt: str) -> str:
    toks = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm.device)
    try:
        with torch.no_grad():
            out = llm.generate(
                **toks,
                max_new_tokens=180,     
                do_sample=False,          
                repetition_penalty=1.05,
                pad_token_id=tok.eos_token_id,
            )
    except AttributeError as e:
        
        if "get_max_length" in str(e):
            with torch.no_grad():
                out = llm.generate(
                    **toks,
                    max_new_tokens=180,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tok.eos_token_id,
                    use_cache=False,
                )
        else:
            raise
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip() if "<|assistant|>" in text else text.strip()


def add_paper(pdf_path: str, paper_id: str) -> int:
    text = read_pdf_text(pdf_path)
    chunks = smart_chunks(text, paper_id)
    p_chunk[paper_id] = chunks
    add_chunks_to_chroma(chunks)
    return len(chunks)

def compare_papers(question: str, paper_a: str, paper_b: str) -> str:
    
    a_hits = axis_topk(question, paper_a, per_axis=axis_k , final_k=top_k)
    b_hits = axis_topk(question, paper_b, per_axis=axis_k , final_k=top_k)
    prompt = make_contrast_prompt(question, a_hits, b_hits, max_ctx_chars=chars, bullets=bullets)
    return generate_answer(prompt)




#   Gradio-APP
state = {"a_id": None, "b_id": None, "a_chunks": 0, "b_chunks": 0}

def do_index(pdf_a, pdf_b):
    if pdf_a is None or pdf_b is None:
        return "Please upload both Paper A and Paper B.", 0, 0
    a_id = os.path.splitext(os.path.basename(pdf_a.name))[0] or "paper_a"
    b_id = os.path.splitext(os.path.basename(pdf_b.name))[0] or "paper_b"
    a_n = add_paper(pdf_a.name, a_id)
    b_n = add_paper(pdf_b.name, b_id)
    state.update({"a_id": a_id, "b_id": b_id, "a_chunks": a_n, "b_chunks": b_n})
    return f"Indexed A:{a_n} chunks, B:{b_n} chunks • Device: {device.upper()} • Chroma: ON", a_n, b_n

def do_compare(question):
    if not state["a_id"] or not state["b_id"]:
        return "Please index both papers first."
    if not question or len(question.strip()) < 3:
        return "Please enter a specific comparison question."
    try:
        ans = compare_papers(question.strip(), state["a_id"], state["b_id"])
        return ans
    except Exception as e:
        return f"Generation error: {e}"




with gr.Blocks(title=" PaperCompare-RAG") as demo:

    gr.Markdown("""
    
    <h1 align="center">PaperCompareRAG — Compares two research papers on training hyperparameters</h1>
    <p align="center"> BGE embeddings, ChromaDB, Phi-3 mini.</p>

    Upload two AI paper PDFs.
    prefer text_based PDFs NOT scans.
    Rename files meaningfully (e.g., BERT.pdf, RoBERTa.pdf) so the app’s per-paper IDs are clear            
    ask question or choose from the examples
    get 4 bullet points
     1. Batch size
     2. Steps/epochs
     3. learning rate
     4. other training details                                          
                                          
    """)

    with gr.Row():
        fa = gr.File(label="Paper A (PDF)", file_types=[".pdf"])
        fb = gr.File(label="Paper B (PDF)", file_types=[".pdf"])
    idx_btn = gr.Button("Index Papers")
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        a_cnt = gr.Number(label="A Chunks", precision=0)
        b_cnt = gr.Number(label="B Chunks", precision=0)
    idx_btn.click(do_index, inputs=[fa, fb], outputs=[status, a_cnt, b_cnt])

    q = gr.Textbox(label="Your question", lines=2, placeholder="ASK Compare fine-tuning batch size, epochs, and learning rate")
    ask_btn = gr.Button("Ask")
    examples = gr.Examples(
    examples=[
        
        "Compare global batch size, total steps/epochs, base learning rate + schedule, and warmup/dropout for each paper's main training run. Extract exact numbers where stated.",
        "On ImageNet-1k base models, compare batch size, training epochs, base LR + schedule (cosine/linear/step), and warmup/dropout. Extract exact numbers where stated.",
        "On COCO main training, compare global batch (images/GPU-GPUs), total iterations/epochs, base LR + step schedule, and warmup/dropout for the primary configuration. Extract exact numbers where stated.",
        "For the main pretraining run, compare effective/global batch size, total optimization steps (or tokens seen), base LR + schedule, and warmup/dropout. Extract exact numbers where stated.",
    ],
    inputs=[q],
    label="Click an example question"
)

    out = gr.Textbox(label="Answer", lines=8)
    ask_btn.click(do_compare, inputs=[q], outputs=[out])

if __name__ == "__main__":
    print(f"Device: {device.upper()} • ChromaDB: ON")
    demo.launch(share=False)
